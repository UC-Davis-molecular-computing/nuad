"""
Shipped with DNA single-stranded tile (SST) sequence designer used in the following publication.
 "Diverse and robust molecular algorithms using reprogrammable DNA self-assembly"
 Woods\*, Doty\*, Myhrvold, Hui, Zhou, Yin, Winfree. (\*Joint first co-authors)

Generally this module processes Python 'ACTG' strings
(as opposed to numpy arrays which are processed by dsd.np).
"""  # noqa

import itertools
import math
import os
import logging
import random
import subprocess as sub
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Sequence, Union, Tuple, List, Dict, Iterable, Optional, cast

os_is_windows = sys.platform == 'win32'

global_thread_pool = ThreadPool()

parameter_set_directory = 'nupack_viennaRNA_parameter_files'

default_vienna_rna_parameter_filename = 'dna_mathews1999.par'  # closer to nupack than dna_mathews2004.par

default_temperature = 37.0
"""Default temperature used to specify a `NUPACK 4 model`_.

.. _NUPACK 4 model: https://piercelab-caltech.github.io/nupack-docs/model/
"""

default_magnesium = 0.0125
default_sodium = 0.05


# unix path must be able to find NUPACK, and NUPACKHOME must be set,
# as described in NUPACK installation instructions.

def _dg_adjust(temperature: float, num_seqs: int) -> float:
    """
    Additive adjustment factor to convert NUPACK's mole fraction units to molar.

    :param temperature: temperature in Celsius
    :param num_seqs: number of sequences
    :return: Additive adjustment factor to convert NUPACK's mole fraction units to molar.
    """
    r = 0.0019872041  # Boltzmann's constant in kcal/mol/K
    water_conc = 55.14  # molar concentration of water at 37 C; ignore temperature dependence, ~5%
    temperature_kelvin = temperature + 273.15  # Kelvin
    # converts from NUPACK mole fraction units to molar units, per association
    adjust = r * temperature_kelvin * math.log(water_conc)
    return adjust * (num_seqs - 1)


@lru_cache(maxsize=10_000)
def pfunc(seqs: Union[str, Tuple[str, ...]],
          temperature: float = default_temperature,
          adjust: bool = True,
          ) -> float:
    """Calls NUPACK's pfunc (http://www.nupack.org/) on a complex consisting of the unique strands in
    seqs, returns energy ("delta G"), i.e., generally a negative number.

    NUPACK version 2 or 3 must be installed and on the PATH.

    :param seqs:
        DNA sequences (list or tuple), whose order indicates a cyclic permutation of the complex
        For one or two sequences, there is only one cyclic permutation, so the order doesn't matter
        in such cases.
    :param temperature:
        temperature in Celsius
    :param adjust:
        whether to adjust from NUPACK mole fraction units to molar units
    :return:
        complex free energy ("delta G") of ordered complex with strands in given cyclic permutation
    """
    if isinstance(seqs, str):
        seqs = (seqs,)
    seqs_on_separate_lines = '\n'.join(seqs)
    permutation = ' '.join(map(str, range(1, len(seqs) + 1)))
    user_input = f'''\
{len(seqs)}
{seqs_on_separate_lines}
{permutation}'''

    command_strs = ['pfunc', '-T', str(temperature), '-multi', '-material', 'dna']

    output, _ = call_subprocess(command_strs, user_input)

    lines = output.split('\n')
    if lines[-4] != "% Free energy (kcal/mol) and partition function:":
        raise NameError('NUPACK output parsing problem')
    dg_str = lines[-3].strip()
    if dg_str.lower() == 'inf':
        # this can occur when two strands have MFE completely unpaired; should be 0 energy
        dg = 0.0
    else:
        dg = float(dg_str)

    if adjust:
        dg += _dg_adjust(temperature, len(seqs))

    return dg


_cached_pfunc4_models = {}


@lru_cache(maxsize=10_000)
def pfunc4(seqs: Union[str, Tuple[str, ...]],
           temperature: float = default_temperature,
           sodium: float = default_sodium,
           magnesium: float = default_magnesium,
           adjust: bool = False,
           ) -> float:
    """
    Calls pfunc from NUPACK 4 (http://www.nupack.org/) on a complex consisting of the unique strands in
    seqs, returns energy ("delta G"), i.e., generally a negative number.

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.

    :param seqs:
        DNA sequences (tuple, or a single DNA sequence),
        whose order indicates a cyclic permutation of the complex.
        For one or two sequences, there is only one cyclic permutation,
        so the order doesn't matter in such cases.
    :param temperature:
        temperature in Celsius
    :param sodium:
        molarity of sodium in moles per liter (Default: 0.05)
    :param magnesium:
        molarity of magnesium in moles per liter (Default: 0.0125)
    :param adjust:
        whether to adjust from NUPACK mole fraction units to molar units
        (was necessary in NUPACK 3, but might not be necessary in NUPACK 4; leaving as an option
        until we know for sure)
    :return:
        complex free energy ("delta G") of ordered complex with strands in given cyclic permutation
    """
    if isinstance(seqs, str):
        seqs = (seqs,)

    try:
        from nupack import pfunc as nupack_pfunc  # type: ignore
        from nupack import Model  # type: ignore
    except ModuleNotFoundError:
        raise ImportError(
            'NUPACK 4 must be installed to use pfunc4. Installation instructions can be found at '
            'https://piercelab-caltech.github.io/nupack-docs/start/.')

    param = (temperature, sodium, magnesium)
    if param not in _cached_pfunc4_models:
        model = Model(celsius=temperature, sodium=sodium, magnesium=magnesium, material='dna')
        _cached_pfunc4_models[param] = model
    else:
        model = _cached_pfunc4_models[param]
    (_, dg) = nupack_pfunc(strands=seqs, model=model)

    if adjust:
        dg += _dg_adjust(temperature, len(seqs))

    return dg


def call_subprocess(command_strs: List[str], user_input: str) -> Tuple[str, str]:
    """
    Calls system command through a subprocess. Assumes running on a POSIX operating system.

    If running on Windows, automatically appends "wsl -e" to start of command to call command
    through Windows subsystem for Linux, so wsl must be installed for this to work:
    `https://docs.microsoft.com/en-us/windows/wsl/install-win10 <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_

    :param command_strs:
        List of command and command line arguments, i.e., to call ``ls -l -a``,
        `command_strs` should be the list ['ls', '-l', '-a'].
    :param user_input:
        Input to give once program is running (i.e., what would user type).
    :return:
        pair of strings (output, error), giving the strings written to stdout and stderr, respectively.
    """
    # When porting the code from python2 to python3 we found an issue with sub.Popen().
    # Passing either of the keyword arguments universal_newlines=True or encoding='utf8'
    # solves the problem for python3.6. For python3.7 (but not 3.6) one can use text=True
    # XXX: Then why are none of those keyword arguments being used here??
    process: Optional[sub.Popen] = None
    command_strs = (['wsl.exe', '-e'] if os_is_windows else []) + command_strs
    try:
        with sub.Popen(command_strs, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE) as process:
            output, stderr = process.communicate(user_input.encode())
    except BaseException as error:
        if process is not None:
            process.kill()
        raise error
    output_decoded = output.decode()
    stderr_decoded = stderr.decode()
    return output_decoded, stderr_decoded


# def rna_duplex_python(seq_pairs: Sequence[Tuple[str, str]],
#                       temperature: float = default_temperature,
#                       negate: bool = False, ) -> List[float]:
#     import RNA
#     (ss, mfe) = RNA.fold('ACGTAGCTGATCGTAGCTAGCTAGCTAGCTAGCTAGCT')
#     print(f'ss = {ss}')
#     print(f'mfe = {mfe}')

# def nupack_duplex_multiple(seq_pairs: Sequence[Tuple[str, str]],
#                            temperature: float = default_temperature,
#                            negate: bool = False) -> List[float]:
#     seqs_list, seq_orders = unique_seqs_in_pairs(seq_pairs)
#     with open('')


def unique_seqs_in_pairs(seq_pairs: Iterable[Tuple[str, str]]) -> Tuple[List[str], Dict[str, int]]:
    """
    :param seq_pairs: iterable of pairs of strings
    :return: list of unique strings in pairs in `seq_pairs` in the order they appear, along with
             dict mapping each string to the order in which it appears in the list
    """
    seq_orders: Dict[str, int] = defaultdict(int)
    seqs_list: List[str] = []
    order = 0
    for pair in seq_pairs:
        for seq in pair:
            if seq not in seq_orders:
                seq_orders[seq] = order
                order += 1
                seqs_list.append(seq)
    return seqs_list, seq_orders


# https://github.com/python/cpython/blob/42336def77f53861284336b3335098a1b9b8cab2/Lib/functools.py#L485
_sentinel = object()
_max_size = 1_000_000
_hits = 0
_misses = 0
_rna_duplex_cache: Dict[Tuple[Tuple[str, str], float, bool, str], float] = {}


def _rna_duplex_single_cacheable(seq_pair: Tuple[str, str], temperature: float, negate: bool,
                                 parameters_filename: str) -> Optional[float]:
    # return None if arguments are not in _rna_duplex_cache,
    # otherwise return cached energy in _rna_duplex_cache
    global _hits
    global _misses
    key = (seq_pair, temperature, negate, parameters_filename)
    result: Union[object, float] = _rna_duplex_cache.get(key, _sentinel)
    if result is _sentinel:
        _misses += 1
        return None
    else:
        _hits += 1
        return cast(float, result)


def rna_duplex_multiple(seq_pairs: Sequence[Tuple[str, str]],
                        logger: logging.Logger = logging.root,
                        temperature: float = default_temperature,
                        negate: bool = False,
                        parameters_filename: str = default_vienna_rna_parameter_filename,
                        # cache: bool = True,
                        cache: bool = False,  # off until I implement LRU queue to bound cache size
                        ) -> List[float]:
    """
    Calls RNAduplex (from ViennaRNA package: https://www.tbi.univie.ac.at/RNA/)
    on a list of pairs, specifically:
    [ (seq1, seq2), (seq2, seq3), (seq4, seq5), ... ]
    where seqi is a string over {A,C,T,G}. Temperature is in Celsius.
    Returns a list (in the same order as seqpairs) of free energies.

    :param seq_pairs: sequence (list or tuple) of pairs of DNA sequences
    :param logger: logger to use for printing error messages
    :param temperature: temperature in Celsius
    :param negate: whether to negate the standard free energy (typically free energies are negative;
                   if `negate` is ``True`` then the return value will be positive)
    :param parameters_filename:
    :param cache:
        Whether to cache results to save on number of sequences to give to RNAduplex.
    :return:
        list of free energies, in the same order as `seq_pairs`
    """
    # print(f'rna_duplex_multiple.lru_cache = {rna_duplex_multiple.cache_info()}')

    # NB: the string NA_parameter_set needs to be exactly the intended filename;
    # e.g. any extra whitespace characters cause RNAduplex to default to RNA parameter set
    # without warning the user!

    # Note that loading parameter set dna_mathews2004.par throws a warning encoded in that parameter set:
    # WARNING: stacking enthalpies not symmetric

    # https://stackoverflow.com/questions/10174211/how-to-make-an-always-relative-to-current-module-file-path

    # fill in cached energies and determine which indices still need to have their energies calculated
    if cache:
        energies = [_rna_duplex_single_cacheable(seq_pair, temperature, negate, parameters_filename)
                    for seq_pair in seq_pairs]
    else:
        energies = [None] * len(seq_pairs)
    idxs_to_calculate = [i for i, energy in enumerate(energies) if energy is None]
    idxs_to_calculate_set = set(idxs_to_calculate)
    seq_pairs_to_calculate = [seq_pair for i, seq_pair in enumerate(seq_pairs) if i in idxs_to_calculate_set]

    full_parameters_filename = os.path.join(os.path.dirname(__file__),
                                            parameter_set_directory, parameters_filename)

    if os_is_windows:
        full_parameters_filename = _fix_filename_windows(full_parameters_filename)

    command_strs: List[str] = ['RNAduplex', '-P', full_parameters_filename, '-T', str(temperature),
                               '--noGU', '−−noconv']

    # DNA sequences to type after RNAduplex starts up
    user_input = '\n'.join(f'{seq_pair[0]}\n{seq_pair[1]}' for seq_pair in seq_pairs_to_calculate) + '\n@\n'

    output, error = call_subprocess(command_strs, user_input)

    if error.strip() != '':
        logger.warning('error from RNAduplex: ', error)
        if error.split('\n')[0] != 'WARNING: stacking enthalpies not symmetric':
            raise ValueError('I will ignore errors about "stacking enthalpies not symmetric", but this '
                             'is a different error that I don\'t know how to handle. Exiting...'
                             f'\nerror:\n{error}')

    energies_to_calculate: List[float] = []
    lines = output.split('\n')
    for line in lines[:-1]:
        energies_to_calculate.append(float(line.split(':')[1].split('(')[1].split(')')[0]))
    if len(lines) - 1 != len(seq_pairs_to_calculate):
        raise ValueError(f'lengths do not match: #lines:{len(lines) - 1} #seqpairs:{len(seq_pairs)}')

    assert len(energies_to_calculate) == len(seq_pairs_to_calculate)

    if negate:
        energies_to_calculate = [-dg for dg in energies_to_calculate]

    # put calculated energies into list to return alongside cached energies
    assert len(idxs_to_calculate) == len(energies_to_calculate)
    for i, energy, seq_pair in zip(idxs_to_calculate, energies_to_calculate, seq_pairs_to_calculate):
        energies[i] = energy
        if cache:
            if _misses < _max_size:
                key = (seq_pair, temperature, negate, parameters_filename)
                _rna_duplex_cache[key] = energy
            else:
                logger.warning(f'WARNING: cache size {_max_size} exceeded; '
                               f'not storing RNAduplex energies in cache anymore.')

    return energies


def _fix_filename_windows(parameters_filename: str) -> str:
    # FIXME: Here's the story: when developing in Windows, os.path will construct the path with Windows
    #  absolute paths. But we need to pass off the computation to wsl.exe (Windows Subsystem for Linux),
    #  which expects Linux-style paths (and has no idea what to do with 'C:\'). So we manually translate
    #  the absolute path. But this is fugly, and we should be not using absolute paths in this way.
    for drive in ['C', 'c', 'D', 'd', 'E', 'e', 'F', 'f']:
        parameters_filename = parameters_filename.replace(f'{drive}:\\', f'/mnt/{drive.lower()}/')
    parameters_filename = parameters_filename.replace('\\', '/')
    return parameters_filename


def rna_cofold_multiple(seq_pairs: Sequence[Tuple[str, str]],
                        logger: logging.Logger = logging.root,
                        temperature: float = default_temperature,
                        negate: bool = False,
                        parameters_filename: str = default_vienna_rna_parameter_filename,
                        ) -> List[float]:
    """
    Calls RNAcofold (from ViennaRNA package: https://www.tbi.univie.ac.at/RNA/)
    on a list of pairs, specifically:
    [ (seq1, seq2), (seq2, seq3), (seq4, seq5), ... ]
    where seqi is a string over {A,C,T,G}. Temperature is in Celsius.
    Returns a list (in the same order as seqpairs) of free energies.

    :param seq_pairs: sequence (list or tuple) of pairs of DNA sequences
    :param logger: logger to use for printing error messages
    :param temperature: temperature in Celsius
    :param parameters_filename:
    :param negate: whether to negate the standard free energy (typically free energies are negative;
                   if `negate` is ``True`` then the return value will be positive)
    :return: list of free energies, in the same order as `seq_pairs`
    """

    # NB: the string NA_parameter_set needs to be exactly the intended filename;
    # e.g. any extra whitespace characters cause RNAduplex to default to RNA parameter set
    # without warning the user!

    # Note that loading parameter set dna_mathews2004.par throws a warning encoded in that parameter set:
    # WARNING: stacking enthalpies not symmetric

    # https://stackoverflow.com/questions/10174211/how-to-make-an-always-relative-to-current-module-file-path
    full_parameters_filename = os.path.join(os.path.dirname(__file__),
                                            parameter_set_directory, parameters_filename)

    if os_is_windows:
        full_parameters_filename = _fix_filename_windows(full_parameters_filename)

    # DNA sequences to type after RNAcofold starts up
    user_input = '\n'.join(seqpair[0] + '&' + seqpair[1] for seqpair in seq_pairs) + '\n@\n'

    command_strs: List[str] = ['RNAcofold', '-P', full_parameters_filename, '-T', str(temperature),
                               '--noGU', '−−noconv', '-p']

    output, stderr = call_subprocess(command_strs, user_input)

    if stderr.strip() != '':
        logger.warning('error from RNAduplex: ', stderr)
        if stderr.split('\n')[0] != 'WARNING: stacking enthalpies not symmetric':
            raise ValueError('I will ignore errors about "stacking enthalpies not symmetric", but this '
                             'is a different error that I don\'t know how to handled. Exiting.')

    lines = output.split('\n')
    dg_list: List[float] = []
    for line in lines[:-1]:
        dg_list.append(-float(line.split(':')[1].split('(')[1].split(')')[0]))
    if len(lines) - 1 != len(seq_pairs):
        raise AssertionError(
            'lengths do not match: #lines:{} #seqpairs:{}'.format(len(lines) - 1, len(seq_pairs)))

    if negate:
        dg_list = [-dg for dg in dg_list]

    return dg_list


_wctable = str.maketrans('ACGTacgt', 'TGCAtgca')


def wc(seq: str) -> str:
    """Return reverse Watson-Crick complement of `seq`."""
    return seq.translate(_wctable)[::-1]


def binding_complement(seq: str, temperature: float = default_temperature, subtract_indv: bool = True,
                       negate: bool = False) -> float:
    """Computes the (partition function) free energy of a strand with its perfect WC complement.

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    seq1 = seq
    seq2 = wc(seq)
    # this is a hack to save time since (seq1,seq2) and (seq2,seq1) are
    #   considered different tuples hence are cached differently by lrucache;
    #   but pfunc is a symmetric function with only two sequences, so it's safe to swap the order
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    association_energy = pfunc((seq1, seq2), temperature, negate)
    if subtract_indv:
        # ddG_reaction == dG(products) - dG(reactants)
        association_energy -= (pfunc(seq1, temperature, negate) + pfunc(seq2, temperature, negate))
    return association_energy


def binding_complement4(seq: str, temperature: float = default_temperature, sodium: float = default_sodium,
                        magnesium: float = default_magnesium, subtract_indv: bool = True,
                        negate: bool = False) -> float:
    """Computes the (partition function) free energy of a strand with its perfect WC complement.

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    seq1 = seq
    seq2 = wc(seq)
    # this is a hack to save time since (seq1,seq2) and (seq2,seq1) are
    #   considered different tuples hence are cached differently by lrucache;
    #   but pfunc is a symmetric function with only two sequences, so it's safe to swap the order
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    association_energy = pfunc4((seq1, seq2), temperature, sodium, magnesium, negate)
    if subtract_indv:
        # ddG_reaction == dG(products) - dG(reactants)
        association_energy -= (pfunc4(seq1, temperature, sodium, magnesium, negate) +
                               pfunc4(seq2, temperature, sodium, magnesium, negate))
    return association_energy


def secondary_structure_single_strand(seq: str, temperature: float = default_temperature,
                                      negate: bool = False) -> float:
    """Computes the (partition function) free energy of single-strand secondary structure.

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    return pfunc((seq,), temperature, negate)


def secondary_structure_single_strand4(
        seq: str, temperature: float = default_temperature, sodium: float = default_sodium,
        magnesium: float = default_magnesium, negate: bool = False) -> float:
    """Computes the (partition function) free energy of single-strand secondary structure.

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    return pfunc4((seq,), temperature, sodium, magnesium, negate)


def binding(seq1: str, seq2: str, temperature: float = default_temperature, negate: bool = False) -> float:
    """Computes the (partition function) free energy of association between two strands.

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    # this is a hack to save time since (seq1,seq2) and (seq2,seq1) are
    #   considered different tuples hence are cached differently by lrucache;
    #   but pfunc is a symmetric function so it's safe to swap the order
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    return pfunc((seq1, seq2), temperature, negate) - (
            pfunc(seq1, temperature, negate) + pfunc(seq2, temperature, negate))


def binding4(seq1: str, seq2: str, *, temperature: float = default_temperature,
             sodium: float = default_sodium, magnesium: float = default_magnesium) -> float:
    """Computes the (partition function) free energy of association between two strands.

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    # this is a hack to save time since (seq1,seq2) and (seq2,seq1) are
    #   considered different tuples hence are cached differently by lrucache;
    #   but pfunc is a symmetric function so it's safe to swap the order
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    return pfunc4((seq1, seq2), temperature, sodium, magnesium) - (
            pfunc4(seq1, temperature, sodium, magnesium) + pfunc4(seq2, temperature, sodium, magnesium))


def random_dna_seq(length: int, bases: Sequence = 'ACTG') -> str:
    """Chooses a random DNA sequence."""
    return ''.join(random.choices(population=bases, k=length))


def domain_equal_strength(seq: str, temperature: float, low: float, high: float) -> bool:
    """test roughly equal strength of domains (according to partition function)

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    dg = binding(seq, wc(seq), temperature)
    return low <= dg <= high


def domain_equal_strength4(seq: str, temperature: float, sodium: float,
                           magnesium: float, low: float, high: float) -> bool:
    """test roughly equal strength of domains (according to partition function)

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    dg = binding4(seq, wc(seq), temperature=temperature, sodium=sodium, magnesium=magnesium)
    return low <= dg <= high


def domain_no_sec_struct(seq: str, temperature: float, individual: float, threaded: bool) -> float:
    """test lack of secondary structure in domains

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    if threaded:
        results = [global_thread_pool.apply_async(secondary_structure_single_strand, args=(s, temperature))
                   for s in (seq, wc(seq))]
        e_seq, e_wcseq = [result.get() for result in results]
        return e_seq <= individual and e_wcseq <= individual
    else:
        seq_sec_struc = secondary_structure_single_strand(seq, temperature)
        seq_wc_sec_struc = secondary_structure_single_strand(wc(seq), temperature)
        return seq_sec_struc <= individual and seq_wc_sec_struc <= individual


def domain_no_sec_struct4(seq: str, temperature: float, sodium: float,
                          magnesium: float, individual: float, threaded: bool) -> float:
    """test lack of secondary structure in domains

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    if threaded:
        results = [global_thread_pool.apply_async(
            secondary_structure_single_strand4, args=(s, temperature, sodium, magnesium))
            for s in (seq, wc(seq))]
        e_seq, e_wcseq = [result.get() for result in results]
        return e_seq <= individual and e_wcseq <= individual
    else:
        seq_sec_struc = secondary_structure_single_strand4(seq, temperature, sodium, magnesium)
        seq_wc_sec_struc = secondary_structure_single_strand4(wc(seq), temperature, sodium, magnesium)
        return seq_sec_struc <= individual and seq_wc_sec_struc <= individual


LOG_ENERGY = False


def log_energy(energy: float) -> None:
    if LOG_ENERGY:
        print(f'{energy:.1f}')


def domain_orthogonal(seq: str, seqs: Sequence[str], temperature: float, orthogonality: float,
                      orthogonality_ave: float = -1, threaded: bool = True) -> bool:
    """test orthogonality of domain with all others and their wc complements

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    if threaded:
        results = [global_thread_pool.apply_async(binding, args=(s, s, temperature)) for s in (seq, wc(seq))]
        energies = [result.get() for result in results]
        if max(energies) > orthogonality:
            return False
    else:
        ss = binding(seq, seq, temperature)
        log_energy(ss)
        if ss > orthogonality:
            return False
        wsws = binding(wc(seq), wc(seq), temperature)
        log_energy(wsws)
        if wsws > orthogonality:
            return False
    energy_sum = 0.0
    for altseq in seqs:
        if threaded:
            results = [global_thread_pool.apply_async(binding, args=(seq1, seq2, temperature))
                       for seq1, seq2 in itertools.product((seq, wc(seq)), (altseq, wc(altseq)))]
            energies = [result.get() for result in results]
            if max(energies) > orthogonality:
                return False
            energy_sum += sum(energies)
        else:
            sa = binding(seq, altseq, temperature)
            log_energy(sa)
            if sa > orthogonality:
                return False
            sw = binding(seq, wc(altseq), temperature)
            log_energy(sw)
            if sw > orthogonality:
                return False
            wa = binding(wc(seq), altseq, temperature)
            log_energy(wa)
            if wa > orthogonality:
                return False
            ww = binding(wc(seq), wc(altseq), temperature)
            log_energy(ww)
            if ww > orthogonality:
                return False
            energy_sum += sa + sw + wa + ww
    if orthogonality_ave > 0:
        energy_ave = energy_sum / (4 * len(seqs)) if len(seqs) > 0 else 0.0
        return energy_ave <= orthogonality_ave
    else:
        return True


def domain_orthogonal4(seq: str, seqs: Sequence[str], temperature: float, sodium: float,
                       magnesium: float, orthogonality: float,
                       orthogonality_ave: float = -1, threaded: bool = True) -> bool:
    """test orthogonality of domain with all others and their wc complements

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.
    """

    def binding_callback(s1: str, s2: str) -> float:
        return binding4(s1, s2, temperature=temperature, sodium=sodium, magnesium=magnesium)

    if threaded:
        results = [
            global_thread_pool.apply_async(binding_callback, args=(s, s))
            for s in (seq, wc(seq))]
        energies = [result.get() for result in results]
        if max(energies) > orthogonality:
            return False
    else:
        ss = binding4(seq, seq, temperature=temperature, sodium=sodium, magnesium=magnesium)
        log_energy(ss)
        if ss > orthogonality:
            return False
        wsws = binding4(wc(seq), wc(seq), temperature=temperature, sodium=sodium, magnesium=magnesium)
        log_energy(wsws)
        if wsws > orthogonality:
            return False
    energy_sum = 0.0
    for altseq in seqs:
        if threaded:
            results = [
                global_thread_pool.apply_async(binding_callback,
                                               args=(seq1, seq2, temperature, sodium, magnesium))
                for seq1, seq2 in itertools.product((seq, wc(seq)), (altseq, wc(altseq)))]
            energies = [result.get() for result in results]
            if max(energies) > orthogonality:
                return False
            energy_sum += sum(energies)
        else:
            sa = binding4(seq, altseq, temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(sa)
            if sa > orthogonality:
                return False
            sw = binding4(seq, wc(altseq), temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(sw)
            if sw > orthogonality:
                return False
            wa = binding4(wc(seq), altseq, temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(wa)
            if wa > orthogonality:
                return False
            ww = binding4(wc(seq), wc(altseq), temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(ww)
            if ww > orthogonality:
                return False
            energy_sum += sa + sw + wa + ww
    if orthogonality_ave > 0:
        energy_ave = energy_sum / (4 * len(seqs)) if len(seqs) > 0 else 0.0
        return energy_ave <= orthogonality_ave
    else:
        return True


def domain_pairwise_concatenated_no_sec_struct(seq: str, seqs: Sequence[str], temperature: float,
                                               concat: float, concat_ave: float = -1,
                                               threaded: bool = True) -> bool:
    """test lack of secondary structure in concatenated domains

    NUPACK version 2 or 3 must be installed and on the PATH.
    """
    #     if hairpin(seq+seq,temperature) > concat: return False
    #     if hairpin(wc(seq)+wc(seq),temperature) > concat: return False
    energy_sum = 0.0
    for altseq in seqs:
        wc_seq = wc(seq)
        wc_altseq = wc(altseq)
        if threaded:
            results = [global_thread_pool.apply_async(secondary_structure_single_strand,
                                                      args=(seq1 + seq2, temperature)) for
                       (seq1, seq2) in
                       [(seq, altseq),
                        (seq, wc_altseq),
                        (wc_seq, altseq),
                        (wc_seq, wc_altseq),
                        (altseq, seq),
                        (wc_altseq, seq),
                        (altseq, wc_seq),
                        (wc_altseq, wc_seq)]]
            energies = [result.get() for result in results]
            #             print len(results)
            #             print 'pair: %s' % [round(e,1) for e in energies]
            if max(energies) > concat:
                return False
            energy_sum += sum(energies)
        else:
            seq_alt = secondary_structure_single_strand(seq + altseq, temperature)
            if seq_alt > concat:
                return False
            seq_wcalt = secondary_structure_single_strand(seq + wc_altseq, temperature)
            if seq_wcalt > concat:
                return False
            wcseq_alt = secondary_structure_single_strand(wc_seq + altseq, temperature)
            if wcseq_alt > concat:
                return False
            wcseq_wcalt = secondary_structure_single_strand(wc_seq + wc_altseq, temperature)
            if wcseq_wcalt > concat:
                return False
            alt_seq = secondary_structure_single_strand(altseq + seq, temperature)
            if alt_seq > concat:
                return False
            alt_wcseq = secondary_structure_single_strand(altseq + wc_seq, temperature)
            if alt_wcseq > concat:
                return False
            wcalt_seq = secondary_structure_single_strand(wc_altseq + seq, temperature)
            if wcalt_seq > concat:
                return False
            wcalt_wcseq = secondary_structure_single_strand(wc_altseq + wc_seq, temperature)
            if wcalt_wcseq > concat:
                return False
            energy_sum += (seq_alt + seq_wcalt + wcseq_alt + wcseq_wcalt +
                           alt_seq + alt_wcseq + wcalt_seq + wcalt_wcseq)
    if concat_ave > 0:
        energy_ave = energy_sum / (8 * len(seqs)) if len(seqs) > 0 else 0.0
        return energy_ave <= concat_ave
    else:
        return True


def domain_pairwise_concatenated_no_sec_struct4(seq: str, seqs: Sequence[str], temperature: float,
                                                sodium: float, magnesium: float,
                                                concat: float, concat_ave: float = -1,
                                                threaded: bool = True) -> bool:
    """test lack of secondary structure in concatenated domains

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    #     if hairpin(seq+seq,temperature) > concat: return False
    #     if hairpin(wc(seq)+wc(seq),temperature) > concat: return False
    energy_sum = 0.0
    for altseq in seqs:
        wc_seq = wc(seq)
        wc_altseq = wc(altseq)
        if threaded:
            results = [global_thread_pool.apply_async(secondary_structure_single_strand4,
                                                      args=(seq1 + seq2, temperature, sodium, magnesium)) for
                       (seq1, seq2) in
                       [(seq, altseq),
                        (seq, wc_altseq),
                        (wc_seq, altseq),
                        (wc_seq, wc_altseq),
                        (altseq, seq),
                        (wc_altseq, seq),
                        (altseq, wc_seq),
                        (wc_altseq, wc_seq)]]
            energies = [result.get() for result in results]
            #             print len(results)
            #             print 'pair: %s' % [round(e,1) for e in energies]
            if max(energies) > concat:
                return False
            energy_sum += sum(energies)
        else:
            seq_alt = secondary_structure_single_strand4(seq + altseq, temperature, sodium, magnesium)
            if seq_alt > concat:
                return False
            seq_wcalt = secondary_structure_single_strand4(seq + wc_altseq, temperature, sodium, magnesium)
            if seq_wcalt > concat:
                return False
            wcseq_alt = secondary_structure_single_strand4(wc_seq + altseq, temperature, sodium, magnesium)
            if wcseq_alt > concat:
                return False
            wcseq_wcalt = secondary_structure_single_strand4(wc_seq + wc_altseq, temperature, sodium,
                                                             magnesium)
            if wcseq_wcalt > concat:
                return False
            alt_seq = secondary_structure_single_strand4(altseq + seq, temperature, sodium, magnesium)
            if alt_seq > concat:
                return False
            alt_wcseq = secondary_structure_single_strand4(altseq + wc_seq, temperature, sodium, magnesium)
            if alt_wcseq > concat:
                return False
            wcalt_seq = secondary_structure_single_strand4(wc_altseq + seq, temperature, sodium, magnesium)
            if wcalt_seq > concat:
                return False
            wcalt_wcseq = secondary_structure_single_strand4(wc_altseq + wc_seq, temperature, sodium,
                                                             magnesium)
            if wcalt_wcseq > concat:
                return False
            energy_sum += (seq_alt + seq_wcalt + wcseq_alt + wcseq_wcalt +
                           alt_seq + alt_wcseq + wcalt_seq + wcalt_wcseq)
    if concat_ave > 0:
        energy_ave = energy_sum / (8 * len(seqs)) if len(seqs) > 0 else 0.0
        return energy_ave <= concat_ave
    else:
        return True


_binaryGCTable = str.maketrans('ACTG', '0101')


def domain_concatenated_no_4gc(seq: str, seqs: Sequence[str]) -> bool:
    """prevent {G,C}^4 under concatenation"""
    for altseq in seqs:
        catseq = altseq + seq + altseq
        strength = catseq.translate(_binaryGCTable)
        if '1111' in strength:
            return False
    return True


def domain_no_4gc(seq: str) -> bool:
    """prevent {G,C}^4"""
    return '1111' not in seq.translate(_binaryGCTable)


def domain_concatenated_no_4g_or_4c(seq: str, seqs: Sequence[str]) -> bool:
    """prevent G^4 and C^4 under concatenation"""
    for altseq in seqs:
        catseq = altseq + seq + altseq
        if 'GGGG' in catseq:
            #             print '|GGGG# seq: %s altseq: %s|' % (seq,altseq)
            return False
        if 'CCCC' in catseq:
            #             print '|CCCC# seq: %s altseq: %s|' % (seq,altseq)
            return False
    return True

# def has_hairpin(seq: str, stem_length: int) -> bool:
#     for i in range(len(seq) - 2 * stem_length - 3):
#         subseq = seq[i:i + stem_length]
#         sub_wc = wc(subseq)
#         #XXX: this doesn't seem right; seq[i + stem_length + 3] is a single character
#         if sub_wc in seq[i + stem_length + 3]:
#             return True
#     return False
#
#
# def domain_concatenated_no_hairpin(seq: str, seqs: Iterable[str], stem_length: int = 5) -> bool:
#     """prevent hairpins of stem length 5 or more"""
#     for altseq in seqs:
#         catseq = altseq + seq
#         if has_hairpin(catseq, stem_length):
#             return False
#         catseq = seq + altseq
#         if has_hairpin(catseq, stem_length):
#             return False
#     return True


# def domain_concatenated_no_hairpin_arr(seq, seqsarr, hairpin=5):
#     '''prevent hairpins of stem length 5 or more'''
#     seqarr = dsd.seq2arr(seq)
#     pairs = all_cats(seqarr, seqsarr)
#     if hairpin > 2 * pairs.shape[1] + 3: return True
#     pairsWC = dsd.wc(pairs)
#     pairs_head = pairs[:, :-hairpin]
#     pairsWC_tail = pairsWC[:, hairpin:]
#     toeplitz = dsd.create_toeplitz(pairs_head.shape[1], hairpin)
#     # XXX: think more carefully about this algorithm
#     raise NotImplementedError()
#     return True


# def log_bad_end(reason: str, log: bool) -> None:
#     if log:
#         sys.stdout.write(reason)
#         sys.stdout.flush()


# # XXX: this function signature is terrble; redo it with keyword arguments, or an encapsulating object
# def nextseq(init_seqs, new_seqs, iterator, temperature, low, high, individual,
#             orthogonality, concat, orthogonality_ave, concat_ave,
#             prevent_4gc, prevent_4g_4c, threaded=True):
#     """Return next sequence from iterator that "gets along" with the sequences
#     already in existing_seqs according to parameters."""
#     all_seqs = init_seqs + new_seqs
#     log = True
#     num_searched = 0
#     sys.stdout.write('.')
#     sys.stdout.flush()
#     # seqsarr = dsd.seqs2arr(existing_seqs)
#     for seq in iterator:
#         num_searched += 1
#         #         sys.stdout.write('.')
#         #         sys.stdout.flush()
#         if wc(seq) in all_seqs:
#             continue
#         if prevent_4gc and not domain_no_4gc(seq):  # domain_concatenated_no4GC(seq,new_seqs):
#             log_bad_end('gc4_', log)
#             continue
#         if not prevent_4gc and (
#                 prevent_4g_4c and (
#                 'GGGG' in seq or 'CCCC' in seq)):  # domain_concatenated_no4Gor4C(seq,new_seqs)):
#             log_bad_end('g4c4_', log)
#             continue
#         # if not domain_concatenated_no_hairpin(seq,existing_seqs,hairpin): continue
#         # if hairpin and not domain_concatenated_no_hairpin_arr(seq,seqsarr,hairpin): continue
#         if not domain_equal_strength(seq, temperature, low, high):
#             log_bad_end('eq_', log)
#             continue
#         if not domain_no_sec_struct(seq, temperature, individual, threaded):
#             log_bad_end('idv_', log)
#             continue
#         if not domain_pairwise_concatenated_no_sec_struct(seq, new_seqs, temperature, concat, concat_ave,
#                                                           threaded):
#             log_bad_end('cat_', log)
#             continue
#         if not domain_orthogonal(seq, all_seqs, temperature, orthogonality, orthogonality_ave, threaded):
#             log_bad_end('orth_', log)
#             continue
#         sys.stdout.write('.\n')
#         sys.stdout.flush()
#         return seq, num_searched
#     raise ValueError('no more sequences to search')

# def learnSL(lengths, lowPF, highPF, temperature, num_samples=100):
#     '''Learn appropriate upper and lower bounds for SantaLucia energy (as
#     calculated by DNASeqList.wcenergies) that preserve "many" sequences whose
#     binding energies according to binding function remain.
#
#     Current algorithm gets sample min and max and chooses lowSL and highSL endpoints
#     to contain the middle two quartiles.
#
#     Originally used minimum variance unbiased estimators for min and max,
#     example 2 here:
#     http://www.d.umn.edu/math/Technical%20Reports/Technical%20Reports%202007-/TR%202010/TR_2010_8.pdf
#
#     but that gave too large a range.'''
#     inrange = 0
#     print('Searching for optimal SantaLucia energy range within binding energy ' \
#           + 'lowSL %.2f and highSL %.2f\n***********************' % (lowPF, highPF))
#     energies = []
#     while inrange < num_samples:
#         sys.stdout.write('.')
#         sys.stdout.flush()
#         seq = random_dna_seq(random.choice(lengths))
#         energyBinding = binding(seq, wc(seq), temperature)
#         if lowPF <= energyBinding <= highPF:
#             inrange += 1
#             energySL = dsd.wcenergy(seq, temperature)
#             energies.append(energySL)
#             # print energySL
#         sys.stdout.write('.')
#         sys.stdout.flush()
#     print()
#     energies.sort()
#     # print [round(e,2) for e in energies]
#     lowerPos = num_samples // 4
#     upperPos = num_samples - lowerPos
#     assert lowerPos < upperPos
#     lowSL = energies[lowerPos]
#     highSL = energies[upperPos]
#     return (lowSL, highSL)
#
#
# def learnPF(seqlists, temperature, num_samples=100):
#     '''Learn appropriate upper and lower bounds for partition function energy (as
#     calculated by sst_dsd.binding(s,wc(s))) that preserve "many" sequences whose
#     SantaLucia binding energies according to dsd.wcenergy function remain.
#
#     Current algorithm gets sample min and max and chooses low and high endpoints
#     to contain the middle two quartiles.
#
#     Originally used minimum variance unbiased estimators for min and max,
#     example 2 here:
#     http://www.d.umn.edu/math/Technical%20Reports/Technical%20Reports%202007-/TR%202010/TR_2010_8.pdf
#
#     but that gave too large a range.'''
#     energies = []
#     for i in range(num_samples):
#         seqlist = random.choice(seqlists)
#         if not (isinstance(seqlist, list) or isinstance(seqlist, dsd.DNASeqList)):
#             raise TypeError('seqlist must be DNASeqList or list, not %s' % seqlist.__class__)
#
#         # elements of seqlists could be either DNASeq objects or lists of strings
#         idx = random.randint(0, len(seqlist) - 1)
#         seq = seqlist[idx]
#         energyPF = binding_complement(seq, temperature)
#         energies.append(energyPF)
#         # print 'energyPF:%.2f energySL:%.2f' % (energyPF,seqlist.wcenergy(idx))
#         # sys.stdout.write('.')
#         # sys.stdout.flush()
#     energies.sort()
#     # print [round(e,2) for e in energies]
#     lowerPos = num_samples // 6
#     upperPos = num_samples - lowerPos
#     assert lowerPos < upperPos
#     highPF = energies[upperPos]
#     lowPF = energies[lowerPos]
#     return (lowPF, highPF)


# def design_domains_10_11(howmany=1, temperature=53.0, lowSL=None, highSL=None,
#                          lowPF=None, highPF=None, individual=1.0,
#                          orthogonality=4.5, concat=3.3,
#                          orthogonality_ave=4.5, concat_ave=3.3,
#                          prevent4GC=False, prevent4G4C=True,
#                          hairpin=0, pr=None, init_seqs: Sequence[str] = (), endGC=False):
#     '''Like design domains but specialized to length 10 and 11 domains.
#     Also iterator uses custom code to start with a small(ish) set of sequences with
#     similar binding energies.
#
#     lowSL and highSL are lower and upper limits on the energy as reported by
#     the SantaLucia nearest neighbor energy model as computed in
#     DNASeqList.wcenergy(idx) (i.e., the energy of the sequence bound to its
#     complement). It should be related to target and spread,
#     which are energies related to the partition function of the ordered complex
#     consisting of the sequence and its complement, but frankly I'm not sure
#     what the relationship is in general. The SL energy is MUCH faster to
#     calculate on all sequences at once, so serves as a fast preliminary filter.
#     '''
#     if (lowSL == None or highSL == None) and (lowPF == None or highPF == None):
#         raise ValueError('At least one of the pairs (lowSL,highSL) or (lowPF,highPF) must be specified.')
#     if lowSL:
#         if not highSL: raise ValueError('lowSL specified but not highSL')
#         print('Using user-specified SantaLucia energy range [%.2f,%.2f]' % (lowSL, highSL))
#     if lowPF:
#         if not highPF: raise ValueError('lowPF specified but not highPF')
#         print('Using user-specified partition energy range [%.2f,%.2f]' % (lowPF, highPF))
#
#     if lowSL == None or highSL == None:
#         print('learning SantaLucia energy')
#         lowSL, spreadSL_ret = learnSL((10, 11), lowPF, highPF, temperature)
#         if not highSL: highSL = spreadSL_ret
#         print('Using learned SantaLucia energy range [%.2f,%.2f]' % (lowSL, highSL))
#         seqs10, seqs11 = prefilter_length_10_11(lowSL, highSL, temperature, endGC)
#     elif lowPF == None or highPF == None:
#         s10, s11 = prefilter_length_10_11(lowSL, highSL, temperature, endGC, convert_to_list=False)
#         print('learning partition energy')
#         lowPF, highPF = learnPF((s10, s11), temperature, num_samples=100)
#         print('Using learned partition energy range [%.2f,%.2f]' % (lowPF, highPF))
#         seqs10 = s10.to_list()
#         seqs11 = s11.to_list()
#     elif lowPF and lowSL and highPF and highSL:
#         seqs10, seqs11 = prefilter_length_10_11(lowSL, highSL, temperature, endGC)
#
#     print('num length-10 seqs found:%d' % len(seqs10))
#     print('num length-11 seqs found:%d' % len(seqs11))
#     random.shuffle(seqs10)
#     random.shuffle(seqs11)
#     new_seqs = []
#     on10 = True
#     iter10 = iter(seqs10)
#     iter11 = iter(seqs11)
#
#     num_total10 = len(seqs10)
#     num_total11 = len(seqs11)
#     num_searched10 = 0
#     num_searched11 = 0
#
#     if pr: pr.enable()
#     raise NotImplementedError('put description of constraint parameters in file name')
#     filename = 'seqs/sequences_%s.txt' % str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
#     with open(filename, 'w') as f:
#         while len(new_seqs) < howmany:
#             iterator = iter10 if on10 else iter11
#             sys.stdout.write(str(len(new_seqs)))
#             start_time = time.time()
#             seq, num_searched = nextseq(init_seqs=init_seqs, new_seqs=new_seqs, iterator=iterator,
#                                         temperature=temperature, lowPF=lowPF, highPF=highPF,
#                                         individual=individual,
#                                         orthogonality=orthogonality, concat=concat,
#                                         orthogonality_ave=orthogonality_ave, concat_ave=concat_ave,
#                                         prevent4GC=prevent4GC, prevent4G4C=prevent4G4C, hairpin=hairpin)
#             tot_time = time.time() - start_time
#             if on10:
#                 num_searched10 += num_searched
#             else:
#                 num_searched11 += num_searched
#             if seq:
#                 new_seqs.append(seq)
#                 print(seq)
#                 print((' time: %5.1f secs' % tot_time))
#                 print((' length 10 searched:  %6d' % num_searched10), end=' ')
#                 print((' length 10 remaining: %6d' % (num_total10 - num_searched10)), end=' ')
#                 print((' length 11 searched:  %6d' % num_searched11), end=' ')
#                 print((' length 11 remaining: %6d' % (num_total11 - num_searched11)))
#                 f.write(seq + '\n')
#                 f.flush()
#             else:
#                 print('Could not find %d sequences matching your criteria' % howmany)
#                 break
#             on10 = not on10
#     if pr: pr.disable()
#     return ([s for s in new_seqs if len(s) == 10],
#             [s for s in new_seqs if len(s) == 11])


# def main():
#     #     init_seqs = ['TTGAGGAGAG',
#     #                  'TGTAGTAGGC',
#     #                  'ATGTTTTGGG',
#     #                  'TTGGTGATTC',
#     #                  'AGTTTGTTGC',
#     #                  'ATAGTGGGAG',
#     #                  'AAGGATGGAC',
#     #                  'TGTAATTGGC',
#     #                  'ATAGGGATGC',
#     #                  'TGAGGGTTAG',
#     #                  'TGAAATGGTC',
#     #                  'TAAGTGGTGG',
#     #                  'TGATGAGGTG',
#     #                  'TGTGTGAGAC',
#     #                  'AAGAGAGGAC',
#     #                  'AGGATTGGAG']
#     init_seqs = ['GCCTACTACA', 'CCATCAAACCA', 'GTCCTACACTT', 'GTCCTCTCTT', 'AAGTGTAGGAC',
#                  'AAGAGAGGAC', 'AGGATTGGAG', 'AGAGATTGTTC', 'CTCCAATCCT', 'GAACAATCTCT',
#                  'GCTACACAATT', 'CACCTCATCA', 'AATTGTGTAGC', 'TGATGAGGTG', 'TGTGTGAGAC',
#                  'TTGAAGAAGAC', 'GTCTCACACA', 'GTCTTCTTCAA', 'CCAACCTATTT', 'GACCATTTCA',
#                  'AAATAGGTTGG', 'TGAAATGGTC', 'TAAGTGGTGG', 'AGTAAGAAGGC', 'CCACCACTTA',
#                  'GCCTTCTTACT', 'CTCACTACATT', 'GCATCCCTAT', 'AATGTAGTGAG', 'ATAGGGATGC',
#                  'TGAGGGTTAG', 'TGGTAAGGAAC', 'CTAACCCTCA', 'GTTCCTTACCA', 'GCTCTTCACAA',
#                  'GTCCATCCTT', 'TTGTGAAGAGC', 'AAGGATGGAC', 'TGTAATTGGC', 'TGGGATAGTAG',
#                  'GCCAATTACA', 'CTACTATCCCA', 'GACTTATCCAA', 'GCAACAAACT', 'TTGGATAAGTC',
#                  'AGTTTGTTGC', 'ATAGTGGGAG', 'AATTAGGTAGC', 'CTCCCACTAT', 'GCTACCTAATT',
#                  'GCAATATCACA', 'CCCAAAACAT', 'TGTGATATTGC', 'ATGTTTTGGG', 'TTGGTGATTC',
#                  'TATTGTTAGGC', 'GAATCACCAA', 'GCCTAACAATA', 'CCCTACAACAA', 'CTCTCCTCAA',
#                  'TTGTTGTAGGG', 'TTGAGGAGAG', 'TGTAGTAGGC', 'TGGTTTGATGG', 'CTCCAATCCT',
#                  'GAACAATCTCT', 'CCTCAAATACA', 'CCATCATCAA', 'TGTATTTGAGG', 'TTGATGATGG',
#                  'TGTGTGAGAC', 'TTGAAGAAGAC', 'CCACCACTTA', 'GCCTTCTTACT', 'GACCTACCATA',
#                  'CCTCAACTCA', 'TATGGTAGGTC', 'TGAGTTGAGG', 'TGAGGGTTAG', 'TGGTAAGGAAC',
#                  'GCCAATTACA', 'CTACTATCCCA', 'GACAACTACCT', 'GCTCAATACA', 'AGGTAGTTGTC',
#                  'TGTATTGAGC', 'ATAGTGGGAG', 'AATTAGGTAGC', 'GAATCACCAA', 'GCCTAACAATA',
#                  'CACTAATCACA', 'CTCTCTACCA', 'TGTGATTAGTG', 'TGGTAGAGAG', 'TGTAGTAGGC',
#                  'TGGTTTGATGG']
#     #     init_seqs=[]
#     s10, s11 = design_domains_10_11(howmany=40, temperature=53.0,
#                                     lowSL=10.0, highSL=10.5,
#                                     # targetPF=9.0,spreadPF=0.5,
#                                     prevent4GC=False, prevent4G4C=True,
#                                     individual=1.0,
#                                     orthogonality=4.0, concat=2.5,
#                                     orthogonality_ave=2.2, concat_ave=1.2,
#                                     init_seqs=init_seqs, endGC=True)
#
#     delim = '*' * 79
#     print(delim)
#     print('Python representation:')
#     print(delim)
#     print('s10 = %s' % s10)
#     print('s11 = %s' % s11)
#     print(delim)
#     print('Sequences delimited by newlines:')
#     print(delim)
#     for s in s10: print(s)
#     for s in s11: print(s)
#
#
# if __name__ == "__main__":
#     main()
