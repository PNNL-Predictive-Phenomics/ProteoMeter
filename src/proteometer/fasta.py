from __future__ import annotations

from typing import TYPE_CHECKING

import Bio.SeqIO as SeqIO

from proteometer.peptide import strip_peptide

if TYPE_CHECKING:
    from collections.abc import Iterable

    from Bio.SeqRecord import SeqRecord


def get_sequences_from_fasta(fasta_file: str) -> list[SeqRecord]:
    """Parses a FASTA file and returns a list of sequence records.

    Args:
        fasta_file (str): Path to the FASTA file containing the sequences.

    Returns:
        list[SeqRecord]: A list of SeqRecord objects representing the parsed sequences.
    """
    with open(fasta_file, "r") as f:
        prot_seq_obj = SeqIO.parse(f, "fasta")
        prot_seqs: list[SeqRecord] = [seq_item for seq_item in prot_seq_obj]  # type: ignore
    return prot_seqs


def get_peptide_start_end_positions_from_sequence(
    peptide: str, protein_sequence: str
) -> tuple[int, int] | None:
    """Finds the first start and end positions of a peptide within a protein sequence (1-indexed).

    For example, BCD in ABCDEFG returns (2, 4).

    Args:
        peptide (str): The peptide sequence to search for. Can contain non-alphabetic characters which will be stripped.
        protein_sequence (str): The full protein sequence in which to search for the peptide.

    Returns:
        tuple[int, int] | None: A tuple of peptide start and end positions, or None if the peptide is not in the sequence.
    """
    peptide_stripped = strip_peptide(peptide)
    start_pos = protein_sequence.find(peptide_stripped)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(peptide_stripped)
    return start_pos + 1, end_pos + 1


def get_peptide_start_end_positions_from_fasta(
    peptides: Iterable[str],
    protein_names: Iterable[str],
    fasta_file: str,
) -> tuple[list[int | None], list[int | None]]:
    """Extracts the start and end positions of peptides from a FASTA file (1-indexed).

    Args:
        peptides (Iterable[str]): An iterable of peptide sequences to search for.
            Can contain non-alphabetic characters which will be stripped.
        protein_names (Iterable[str]): An iterable of protein names corresponding to each peptide.
            The protein name should match the identifier in the FASTA file (up to the first `|` character).
        fasta_file (str): Path to the FASTA file containing the protein sequences.

    Returns:
        tuple[list[int | None], list[int | None]]: Two lists containing the start and end positions of peptides, or None if not found.
    """
    sequences = get_sequences_from_fasta(fasta_file)
    prot_seq_dict = {
        seq.id.split("|")[0] if seq.id else "": str(seq.seq) for seq in sequences
    }
    if "" in prot_seq_dict:
        del prot_seq_dict[""]
    start_positions: list[int | None] = []
    end_positions: list[int | None] = []
    for peptide, prot_name in zip(peptides, protein_names, strict=True):
        prot_seq = prot_seq_dict.get(prot_name)
        if prot_seq is None:
            start_positions.append(None)
            end_positions.append(None)
            continue
        start_end_pos = get_peptide_start_end_positions_from_sequence(peptide, prot_seq)
        if start_end_pos is None:
            start_positions.append(None)
            end_positions.append(None)
        else:
            start_positions.append(start_end_pos[0])
            end_positions.append(start_end_pos[1])
    return start_positions, end_positions
