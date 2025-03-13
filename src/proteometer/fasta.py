# type: ignore
import Bio.SeqIO as SeqIO


def get_sequences_from_fasta(fasta_file):
    prot_seq_obj = SeqIO.parse(fasta_file, "fasta")
    prot_seqs = [seq_item for seq_item in prot_seq_obj]
    return prot_seqs
