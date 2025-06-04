from __future__ import annotations

from typing import TYPE_CHECKING

import Bio.SeqIO as SeqIO

if TYPE_CHECKING:
    from Bio.SeqRecord import SeqRecord


def get_sequences_from_fasta(fasta_file: str) -> list[SeqRecord]:
    """Parses a FASTA file and returns a list of sequence records.

    Args:
        fasta_file (str): Path to the FASTA file containing the sequences.

    Returns:
        list[SeqRecord]: A list of SeqRecord objects representing the parsed sequences.
    """
    with open(fasta_file, "r") as f:
        prot_seq_obj = SeqIO.parse(f, "fasta")  # type: ignore
        prot_seqs: list[SeqRecord] = [seq_item for seq_item in prot_seq_obj]  # type: ignore
    return prot_seqs
