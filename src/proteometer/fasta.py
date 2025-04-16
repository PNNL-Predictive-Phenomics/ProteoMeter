from __future__ import annotations

from typing import TYPE_CHECKING

import Bio.SeqIO as SeqIO

if TYPE_CHECKING:
    from Bio.SeqRecord import SeqRecord


def get_sequences_from_fasta(fasta_file: str) -> list[SeqRecord]:
    prot_seq_obj = SeqIO.parse(fasta_file, "fasta")  # type: ignore
    prot_seqs: list[SeqRecord] = [seq_item for seq_item in prot_seq_obj]  # type: ignore
    return prot_seqs
