"""
Shipped with DNA single-stranded tile (SST) sequence designer used in the following publication.
 "Diverse and robust molecular algorithms using reprogrammable DNA self-assembly"
 Woods\*, Doty\*, Myhrvold, Hui, Zhou, Yin, Winfree. (\*Joint first co-authors)

Generally this module processes Python 'ACTG' strings
(as opposed to numpy arrays which are processed by dsd.np).
"""  # noqa
import collections
import itertools
import os
import logging
import random
import subprocess as sub
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Sequence, Union, Tuple, List, Dict, Iterable, Optional, cast, Deque

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

_cached_pfunc4_models = {}


@lru_cache(maxsize=10_000)
def pfunc(seqs: Union[str, Tuple[str, ...]],
          temperature: float = default_temperature,
          sodium: float = default_sodium,
          magnesium: float = default_magnesium,
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
_rna_duplex_cache: Dict[Tuple[Tuple[str, str], float, str], float] = {}
_rna_duplex_queue: Deque[Tuple[Tuple[str, str], float, str]] = collections.deque(maxlen=_max_size)


def _rna_duplex_single_cacheable(seq_pair: Tuple[str, str], temperature: float,
                                 parameters_filename: str) -> Optional[float]:
    # return None if arguments are not in _rna_duplex_cache,
    # otherwise return cached energy in _rna_duplex_cache
    key = (seq_pair, temperature, parameters_filename)
    result: Union[object, float] = _rna_duplex_cache.get(key, _sentinel)
    if result is _sentinel:
        return None
    else:
        result_float: float = cast(float, result)
        return result_float


def rna_duplex_multiple(seq_pairs: Sequence[Tuple[str, str]],
                        logger: logging.Logger = logging.root,
                        temperature: float = default_temperature,
                        parameters_filename: str = default_vienna_rna_parameter_filename,
                        cache: bool = True,
                        # cache: bool = False,  # off until I implement LRU queue to bound cache size
                        ) -> List[float]:
    """
    Calls RNAduplex (from ViennaRNA package: https://www.tbi.univie.ac.at/RNA/)
    on a list of pairs, specifically:
    [ (seq1, seq2), (seq2, seq3), (seq4, seq5), ... ]
    where seqi is a string over {A,C,T,G}. Temperature is in Celsius.
    Returns a list (in the same order as seqpairs) of free energies.

    :param seq_pairs:
        sequence (list or tuple) of pairs of DNA sequences
    :param logger:
        logger to use for printing error messages
    :param temperature:
        temperature in Celsius
    :param parameters_filename:
        name of parameters file for NUPACK
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
    print(f'size of queue: {len(_rna_duplex_queue)}')
    if cache:
        energies = [_rna_duplex_single_cacheable(seq_pair, temperature, parameters_filename)
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

    # put calculated energies into list to return alongside cached energies
    assert len(idxs_to_calculate) == len(energies_to_calculate)
    for i, energy, seq_pair in zip(idxs_to_calculate, energies_to_calculate, seq_pairs_to_calculate):
        energies[i] = energy
        if cache:
            key = (seq_pair, temperature, parameters_filename)
            _rna_duplex_cache[key] = energy

            # clear out oldest cache key if _rna_duplex_queue is full
            if len(_rna_duplex_queue) == _rna_duplex_queue.maxlen:
                lru_item = _rna_duplex_queue[0]
                del _rna_duplex_cache[lru_item]

            _rna_duplex_queue.append(key)

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
                        parameters_filename: str = default_vienna_rna_parameter_filename,
                        ) -> List[float]:
    """
    Calls RNAcofold (from ViennaRNA package: https://www.tbi.univie.ac.at/RNA/)
    on a list of pairs, specifically:
    [ (seq1, seq2), (seq2, seq3), (seq4, seq5), ... ]
    where seqi is a string over {A,C,T,G}. Temperature is in Celsius.
    Returns a list (in the same order as seqpairs) of free energies.

    :param seq_pairs:
        sequence (list or tuple) of pairs of DNA sequences
    :param logger:
        logger to use for printing error messages
    :param temperature:
        temperature in Celsius
    :param parameters_filename:
        name of NUPACK parameters file
    :return:
        list of free energies, in the same order as `seq_pairs`
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

    return dg_list


_wctable = str.maketrans('ACGTacgt', 'TGCAtgca')


def wc(seq: str) -> str:
    """Return reverse Watson-Crick complement of `seq`."""
    return seq.translate(_wctable)[::-1]


def binding_complement(seq: str, temperature: float = default_temperature, sodium: float = default_sodium,
                       magnesium: float = default_magnesium, subtract_indv: bool = True) -> float:
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
    association_energy = pfunc((seq1, seq2), temperature, sodium, magnesium)
    if subtract_indv:
        # ddG_reaction == dG(products) - dG(reactants)
        association_energy -= (pfunc(seq1, temperature, sodium, magnesium) +
                               pfunc(seq2, temperature, sodium, magnesium))
    return association_energy


def secondary_structure_single_strand(
        seq: str, temperature: float = default_temperature, sodium: float = default_sodium,
        magnesium: float = default_magnesium) -> float:
    """Computes the (partition function) free energy of single-strand secondary structure.

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    return pfunc((seq,), temperature, sodium, magnesium)


def binding(seq1: str, seq2: str, *, temperature: float = default_temperature,
            sodium: float = default_sodium, magnesium: float = default_magnesium) -> float:
    """Computes the (partition function) free energy of association between two strands.

    NUPACK 4 must be installed. Installation instructions can be found at https://piercelab-caltech.github.io/nupack-docs/start/.
    """
    # this is a hack to save time since (seq1,seq2) and (seq2,seq1) are
    #   considered different tuples hence are cached differently by lrucache;
    #   but pfunc is a symmetric function so it's safe to swap the order
    if seq1 > seq2:
        seq1, seq2 = seq2, seq1
    return pfunc((seq1, seq2), temperature, sodium, magnesium) - (
            pfunc(seq1, temperature, sodium, magnesium) + pfunc(seq2, temperature, sodium, magnesium))


def random_dna_seq(length: int, bases: Sequence = 'ACTG') -> str:
    """Chooses a random DNA sequence."""
    return ''.join(random.choices(population=bases, k=length))


LOG_ENERGY = False


def log_energy(energy: float) -> None:
    if LOG_ENERGY:
        print(f'{energy:.1f}')


def domain_orthogonal(seq: str, seqs: Sequence[str], temperature: float, sodium: float,
                      magnesium: float, orthogonality: float,
                      orthogonality_ave: float = -1, threaded: bool = True) -> bool:
    """test orthogonality of domain with all others and their wc complements

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.
    """

    def binding_callback(s1: str, s2: str) -> float:
        return binding(s1, s2, temperature=temperature, sodium=sodium, magnesium=magnesium)

    if threaded:
        results = [
            global_thread_pool.apply_async(binding_callback, args=(s, s))
            for s in (seq, wc(seq))]
        energies = [result.get() for result in results]
        if max(energies) > orthogonality:
            return False
    else:
        ss = binding(seq, seq, temperature=temperature, sodium=sodium, magnesium=magnesium)
        log_energy(ss)
        if ss > orthogonality:
            return False
        wsws = binding(wc(seq), wc(seq), temperature=temperature, sodium=sodium, magnesium=magnesium)
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
            sa = binding(seq, altseq, temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(sa)
            if sa > orthogonality:
                return False
            sw = binding(seq, wc(altseq), temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(sw)
            if sw > orthogonality:
                return False
            wa = binding(wc(seq), altseq, temperature=temperature, sodium=sodium, magnesium=magnesium)
            log_energy(wa)
            if wa > orthogonality:
                return False
            ww = binding(wc(seq), wc(altseq), temperature=temperature, sodium=sodium, magnesium=magnesium)
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
            results = [global_thread_pool.apply_async(secondary_structure_single_strand,
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
            seq_alt = secondary_structure_single_strand(seq + altseq, temperature, sodium, magnesium)
            if seq_alt > concat:
                return False
            seq_wcalt = secondary_structure_single_strand(seq + wc_altseq, temperature, sodium, magnesium)
            if seq_wcalt > concat:
                return False
            wcseq_alt = secondary_structure_single_strand(wc_seq + altseq, temperature, sodium, magnesium)
            if wcseq_alt > concat:
                return False
            wcseq_wcalt = secondary_structure_single_strand(wc_seq + wc_altseq, temperature, sodium,
                                                            magnesium)
            if wcseq_wcalt > concat:
                return False
            alt_seq = secondary_structure_single_strand(altseq + seq, temperature, sodium, magnesium)
            if alt_seq > concat:
                return False
            alt_wcseq = secondary_structure_single_strand(altseq + wc_seq, temperature, sodium, magnesium)
            if alt_wcseq > concat:
                return False
            wcalt_seq = secondary_structure_single_strand(wc_altseq + seq, temperature, sodium, magnesium)
            if wcalt_seq > concat:
                return False
            wcalt_wcseq = secondary_structure_single_strand(wc_altseq + wc_seq, temperature, sodium,
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
