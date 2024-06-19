# -*- coding: utf-8 -*-
"""
Module to apply lmfit to large datasets.

Can run in multiple subprocesses to speed up fits. Python does not have true
multithreading, so we use subprocesses. Forking a process takes about half
a second, so only makes sense for long calculations.

We trade some of lmfit's explicitness in having objects for each parameter
for simpler data structures because this should run on huge datasets.

Example usage including caching:

```
if (cache := cache_load(fname)) is not None:
    pbest, pbesterr = cache
else:
    pbest, pbesterr = lmfit_multiprocess(
        Transmissionmodel, Sdata.view(float), f=fVNA/1e9, usecpu=6, p0=p0)
    pbest, pbesterr = cache_pbest(fname, pbest, pbesterr)
```
"""

import os
import time
import numpy as np
from uncertainties import ufloat

# TODO https://stackoverflow.com/a/47428575 but doesn't seem to work
# tqdm >= 4.27.0
#from tqdm.auto import tqdm
from tqdm.notebook import tqdm

# Need multiprocess, fork of multiprocessing due to
# https://stackoverflow.com/a/65001152
from multiprocess import Process, Queue, cpu_count


def lmfit_many(lmmodel, data, p0=None, print_report=False, **hints):
    """
    Fit lmmodel along last axis of `data`.

    Returns both the lmfit ModelResult and a numpy.ndarray for each
    parameter name.

    Exceptions in fitting are silently printed to stdout.
    The returned data will be numpy.nan for this data.

    TODO: don't use ufloat but return separate arrays for best value and
    uncertainty. Cleaner, faster, smaller.

    Parameters
    ----------
    lmmodel : lmfit.model.Model
        The model to fit
    data : numpy.ndarray
        Data to fit. Last axis will be first argument of `lmfit.model.Model.fit()`
    p0 : lmfit.parameter.Parameters, optional
        Dictionary of parameters to use in fit. The default is None.
    print_report : bool, optional
        If true will print the lmfit `fit_report()` to stdout.
        The default is False.
    **hints :
        Additional keyword arguments will be passed directly to the model's fit
        function.

    Returns
    -------
    lmres : list[tuple[tuple[int], lmfit.model.ModelResult]]
        `lmfit.model.ModelResult` for each fit.
        List of 2-tuples of (idx, res). `idx` is the a tuple indexing the dataset
        in `data`. `res` is the `ModelResult`.
    pbest : dict[str: numpy.ndarray of uncertainties.ufloat]
        dict with parameter names as keys.
    exceptions : list[tuple[tuple[int], Exception]]
        Exceptions that occured during fit.
    """
    lmres = []
    outshape = data.shape[:-1]
    pbest = {pname: np.full(outshape, ufloat(np.nan, np.nan))
             for pname in lmmodel.param_names}
    if isinstance(p0, np.ndarray):
        p0 = np.broadcast_to(p0, outshape)
    exceptions = []
    for idx in np.ndindex(data.shape[:-1]):
        try:
            res = lmmodel.fit(
                data[idx],
                p0[idx] if isinstance(p0, np.ndarray) else p0,
                **hints)
            if print_report:
                print(res.fit_report(show_correl=False))
            lmres.append((idx, res))
            params = res.params
            if not res.success: continue
            for pname in lmmodel.param_names:
                pbest[pname][idx] = ufloat(params[pname]._val,
                                           params[pname].stderr or np.nan)
        except Exception as e:
            print("EXCEPTION:", idx, repr(e))
            exceptions.append((idx, e))
    return lmres, pbest, exceptions


def _fit_many_subprocess_p0dict(procid, results, progress, lmmodel, subidxs, data, p0, hints):
    """
    Helper function for multiprocess fitting. This function is executed in
    subprocess by `lmfit_multiprocess()`.

    Prints info about subprocess to stdout.

    Parameters
    ----------
    procid : int
        Process ID (not real system process ID)
    results : multiprocess.Queue
        Queue to communicate results. Queues one object when subprocess
        finishes holding procid, best fit values and uncertainties.
    progress : multiprocess.Queue
        Queue to communicate progress. Queues integers from 0 to number of
        fits in subprocess.
    lmmodel : lmfit.model.Model
        Model to fit.
    subidxs : list of tuples
        List of subidxs in `data` to fit.
    data : numpy.ndarray
        The data.
    p0 : dict of string to numpy.ndarray
        Initial guesses for each parameter.
    hints : dict
        Extra keyword arguments to the `lmfit.model.Model.fit()` function.
    """
    import numpy as np
    print(f"(Subprocess {procid} started.)")

    pbest = {pname: np.full(len(subidxs), np.nan)
             for pname in lmmodel.param_names}
    pbesterr = {pname: np.full(len(subidxs), np.nan)
             for pname in lmmodel.param_names}

    for i in range(len(subidxs)):
        if i % 100 == 0:
            progress.put(i, block=False)
        try:
            params = lmmodel.make_params(**{key: p0[key][i] for key in p0})
            res = lmmodel.fit(data[i], params, **hints)
            if not res.success: continue
            params = res.params
            #print(procid, i, res.fit_report())
            for pname in lmmodel.param_names:
                pbest[pname][i] = params[pname]._val
                pbesterr[pname][i] = params[pname].stderr or np.nan
        except Exception as e:
            print(f"Exception in subprocess {procid} at data idx {repr(subidxs[i])}:", repr(e))
            #exceptions.append((subidxs[i], e)) # TODO

    results.put((procid, pbest, pbesterr))
    print(f"(Subprocess {procid} finished.)")


def lmfit_multiprocess(lmmodel, data, p0, usecpu=None, **hints):
    """
    Fit `lmmodel` along last axis of `data` in multiple subprocesses.

    Uses tqdm to show progress bar.

    Prints minimal info to stdout.
    Exceptions during fit are ignored (but printed to stdout).

    Parameters
    ----------
    lmmodel : lmfit.model.Model
        Model to fit.
    data : numpy.ndarray
        The data to fit.
    p0 : dict[str: numpy.ndarray]
        Initial parameters for each fit. Will be broadcast to match the data
        shape (except last axis, along which the fit runs).
    usecpu : int or None, optional
        Number of subprocesses to spawn. If none will use number of CPUs of the
        system. The default is None.
    **hints : TYPE
        Additional keyword arguments to `lmfit.model.Model.fit()`.
        For example the independent variable in the fit.

    Returns
    -------
    pbest : dict of numpy.ndarray
        Dictionary with best fit values, keys are parameter values.
        Values are `numpy.nan` if fit failed
    pbesterr : dict of numpy.ndarray
        Uncertainties in same format
    """

    print("Preparing data")
    # this is a n-dim index for all but last axis
    # used to identify errors and exceptions
    idxs = list(np.ndindex(data.shape[:-1]))
    assert len(idxs) == np.prod(data.shape[:-1])
    n = len(idxs)

    # reshape to flatten all but last axis
    calcshape = (n, data.shape[-1])
    flatdata = data.reshape(calcshape)

    print("Preparing p0")
    # broadcast p0, then flatten
    flatp0 = {}
    for key in p0:
        flatp0[key] = np.broadcast_to(p0[key], data.shape[:-1]).reshape(n)

    print("Starting processes")
    results = Queue() # used to get results from subprocess

    # Distribute and start subprocesses
    cpucount = cpu_count() if usecpu is None else usecpu
    npercpu = int(np.ceil(n / cpucount))
    procs = []
    progress = [] # [(Queue, tqdm)]

    # Start and wait for processes to join in an interruptible way.
    try:
        for i in range(cpucount):
            start, stop = i*npercpu, (i+1)*npercpu
            subidxs = idxs[start:stop]
            if not subidxs:
                break
            subp0 = {key: flatp0[key][start:stop] for key in flatp0}
            prog = Queue()
            #print(f"Starting process {i} with {len(subidxs)} fits.")
            p = Process(target=_fit_many_subprocess_p0dict, args=(
                i, results, prog, lmmodel, subidxs, flatdata[start:stop], subp0, hints))
            procs.append(p)
            progress.append((prog, tqdm(total=len(subidxs), position=i)))
            p.start()

        # Processes are finished when they have supplied a result.
        res = []
        while len(res) < len(procs):
            while not results.empty():
                res.append(results.get())
                print(f"Have {len(res)} results.")
            for i, (prog, progbar) in enumerate(progress):
                if not prog.empty():
                    while not prog.empty():
                        j = prog.get()
                    progbar.update(j-progbar.n)
            time.sleep(0.05)

        for i, p in enumerate(procs):
            p.join() # should be immediate
            if p.exitcode != 0:
                print(f"Process {i} exited with non-zero code: {p.exitcode}")
    except KeyboardInterrupt as e:
        print("Interrupted, killing subprocesses")
        for p in procs:
            p.kill()
        raise e

    res.sort() # sorts by procid i

    # Concat and reshape results
    pbest = {}
    pbesterr = {}
    for pname in lmmodel.param_names:
        pres = np.concatenate([r[1][pname] for r in res])
        preserr = np.concatenate([r[2][pname] for r in res])
        pbest[pname] = pres.reshape(data.shape[:-1])
        pbesterr[pname] = preserr.reshape(data.shape[:-1])

    return pbest, pbesterr


def cache_fpath(fname: str, suffix: str = '_fit', cachedir: str = './') -> str:
    if not fname.endswith(suffix+'.npz'):
        fname = fname + suffix + '.npz'
    return os.path.join(cachedir, fname)


def cache_load(fname, suffix='_fit', cachedir='./', printinfo=True):
    fpath = cache_fpath(fname, suffix, cachedir)
    if not os.path.exists(fpath):
        if printinfo:
            print("Cache miss for ", fpath)
        return None
    if printinfo:
        print("Loading ", fpath)
    cache = np.load(fpath, allow_pickle=True)
    return cache['pbest'][()], cache['pbesterr'][()]


def cache_pbest(fname, pbest, pbesterr=None, suffix='_fit', cachedir='./', printinfo=True):
    if not os.path.exists(cachedir):
        if printinfo:
            print("Creating cache directory ", cachedir)
        os.mkdir(cachedir)
    fpath = cache_fpath(fname, suffix, cachedir)
    np.savez_compressed(fpath, pbest=pbest, pbesterr=pbesterr)
    print("Cached to ", fpath)
    return pbest, pbesterr
