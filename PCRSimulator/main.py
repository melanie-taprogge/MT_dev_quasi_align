# pip3 install psycopg2-binary
# pip3 install qrng
# pip3 install torch torchvision
# pip3 install ray

from abc import ABC, abstractmethod
import psycopg2
import ray
import qrng
import torch
import torchvision
from Bio import SeqIO
import argparse


@ray.remote
class GenomeDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenomeDatabase, cls).__new__(cls)
            cls._instance.genomes = {}
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, fasta_path):
        if not self._initialized:
            self._initialized = True
            with open(fasta_path, "r") as file:
                for record in SeqIO.parse(file, "fasta"):
                    self.genomes[record.id] = record.seq

    def get_genome(self, genome_id):
        return self.genomes.get(genome_id)


# todo: rewrite using BioPython
def read_primers_from_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    primers = {}
    for i in range(0, len(lines), 2):
        header = lines[i].strip().replace(">", "")
        sequence = lines[i + 1].strip()
        primers[header] = sequence

    return primers


# def read_genomes_from_fasta(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     genomes = {}
#     genome_name = ""
#     genome_sequence = ""
#     for line in lines:
#         if line.startswith(">"):
#             if genome_name and genome_sequence:  # save the previous genome
#                 genomes[genome_name] = genome_sequence
#                 genome_sequence = ""
#             genome_name = line.strip().replace(">", "")
#         else:
#             genome_sequence += line.strip()
#
#     # save the last genome
#     if genome_name and genome_sequence:
#         genomes[genome_name] = genome_sequence
#
#     return genomes


# todo: download series of quantum random numbers and fetching them locally
def get_quantum_random_numbers(n=1, lower_bound=0, upper_bound=100):
    """
    Fetches 'n' quantum random integers between 'lower_bound' and 'upper_bound'.
    """
    integers = qrng.get_data(data_type="uint16", array_length=n)
    return [(int(i) % (upper_bound - lower_bound + 1)) + lower_bound for i in integers]


class DNAMolecule:
    def __init__(self, forward_strand, reverse_strand):
        self.forward_strand = forward_strand  # the 5' to 3' strand
        self.reverse_strand = reverse_strand  # the complementary 3' to 5' strand
        self.is_double_stranded = True

    def denature(self):
        # A probability of incomplete denaturation
        if random.random() < 0.95:  # 95% chance of denaturation
            self.is_double_stranded = False

    def anneal(self, primer, current_temp):
        # Calculate the effect of temperature on binding
        temp_factor = 1 - abs(primer.optimal_temp - current_temp) / primer.optimal_temp

        # Check for binding with mismatches on forward strand
        if self._check_binding(self.forward_strand, primer, temp_factor):
            self.is_double_stranded = True
            return True
        # Check for binding with mismatches on reverse strand
        elif self._check_binding(self.reverse_strand, primer, temp_factor):
            self.is_double_stranded = True
            return True
        return False

    def _check_binding(self, strand, primer, temp_factor):
        # Simple mismatch count - can be replaced with a more complex algorithm
        mismatch_count = sum(1 for a, b in zip(strand, primer.sequence) if a != b)
        if mismatch_count <= primer.mismatch_tolerance and temp_factor > 0.5:
            return True
        return False


class Primer:
    def __init__(self, sequence, tm, gc_content, position, specificity=None):
        """
        :param sequence: The nucleotide sequence of the primer.
        :param tm: The melting temperature of the primer.
        :param gc_content: The GC content of the primer.
        :param position: A tuple indicating the start and end position of the primer in the reference sequence.
        :param specificity: An optional parameter that can be used to store data about the specificity of the primer.
        """
        self.sequence = sequence
        self.tm = tm
        self.gc_content = gc_content
        self.position = position
        self.specificity = specificity
        self.is_bound = False

    def display(self):
        """Displays basic primer information."""
        print(f"Primer Sequence: {self.sequence}")
        print(f"Melting Temperature (Tm): {self.tm}°C")
        print(f"GC Content: {self.gc_content}%")
        print(f"Position: {self.position[0]}-{self.position[1]}")

    def bind(self, genome):
        if genome.is_binding_site_present(self.sequence):
            # Handle binding logic
            pass
        else:
            # Handle failed binding logic
            pass

    def denature(self):
        self.is_bound = False


class Genome:
    def __init__(self, sequence: str, species: str, annotations=None):
        """
        Initializes a new Genome instance.

        :param sequence: The nucleotide sequence of the genome (usually A, T, C, G).
        :param species: The species to which this genome belongs.
        :param annotations: A list of annotations or metadata associated with the genome.
        """
        self.sequence = sequence
        self.species = species
        self.annotations = annotations if annotations else []

    def get_subsequence(self, start: int, end: int) -> str:
        """
        Extracts a subsequence from the genome.

        :param start: Start position (0-indexed).
        :param end: End position (exclusive).
        :return: A substring representing the subsequence.
        """
        return self.sequence[start:end]

    def add_annotation(self, annotation: str):
        """
        Adds an annotation to the genome.

        :param annotation: The annotation string to add.
        """
        self.annotations.append(annotation)

    def get_complementary_sequence(self, seq):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join([complement[base] for base in seq])

    # todo: incorporate mismatch tolerance
    def is_binding_site_present(self, primer):
        return primer in self.sequence or self.get_complementary_sequence(primer) in self.sequence

    def __len__(self):
        """
        Returns the length of the genome.

        :return: Length of the genome.
        """
        return len(self.sequence)

    def __repr__(self):
        """
        Provides a string representation for the genome.

        :return: A string representation of the genome.
        """
        return f"<Genome (species: {self.species}, length: {len(self)})>"


class Polymerase:
    def __init__(self, name, elongation_speed, temperature_optimum, proofreading_efficiency=0.0, amplifiable_range=(0, 1000), inhibitor_tolerance):
        self.name = name
        self.elongation_speed = elongation_speed  # nucleotides per second
        self.temperature_optimum = temperature_optimum  # optimum temperature for activity
        # todo: proofreading mechanisms - exonuclease-mediated, intrinsic selectivity for correct nucleotides, its response to mismatches, and other structural and functional properties
        self.proofreading_efficiency = proofreading_efficiency  # 0 (none) to 1 (perfect)
        # todo: optimal_range
        self.amplifiable_range = amplifiable_range
        # todo: make more detailed (different types of inhib., etc.)
        self.inhibitor_tolerance = inhibitor_tolerance # float between 0.0 and 1.0, with 0.0 meaning no tolerance at all and 1.0 being highest tolerance

    def can_amplify(self, template_strand_length):

        return self.amplifiable_range[0] <= template_strand_length <= self.amplifiable_range[1]

    def can_tolerate_inhibitors(self, inhibitor_concentration):

        return self.inhibitor_tolerance >= inhibitor_concentration

    def is_within_optimal_temp(self, temp):

        return self.optimal_temp_range[0] <= temp <= self.optimal_temp_range[1]

    def synthesize_dna(self, template_strand):

        #todo: model proofreading and mismatches
        return template_strand.copy()

    def get_error_rate(self):

        # todo: more complex calculation
        # todo: use AI/ML to mine empirical data (papers, datasets) to get more realistic information about this relationship
        base_error_rate = 0.01
        adjusted_error_rate = base_error_rate * (1 - self.proofreading_efficiency)
        return adjusted_error_rate


# class PCR_Condition:
#     pass
#
#
# class AmplificationResult:
#     pass


#class AlignmentStrategy:
#    pass


# class ClassificationStrategy(ABC):
#
#     @abstractmethod
#     def classify(self, sequence_data):
#         pass

# class MSCStrategy(ClassificationStrategy):
#     pass
#
#
# class SpecificityTester:
#     pass
#
#
# class PrimerPair:
#     pass
#
#
# class Database:
#     pass
#
#
# class EventListener:
#     pass

from abc import ABC, abstractmethod

# Strategy interface for primer specificity check
class PrimerSpecificityStrategy(ABC):
    @abstractmethod
    def test_specificity(self, primer, metagenomic_data):
        pass


class BasicSpecificityTest(PrimerSpecificityStrategy):
    def test_specificity(self, primer, metagenomic_data):
        pass


class AdvancedSpecificityTest(PrimerSpecificityStrategy):
    def test_specificity(self, primer, metagenomic_data):
        pass


class PCR:
    def __init__(self, primers, genomes, dna_molecules):
        self.primers = primers
        self.genomes = genomes
        self.dna_molecules = dna_molecules
        self.observers = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, event):
        for observer in self.observers:
            observer.update(event)

    def denature(self):
        for dna_molecule in self.dna_molecules:
            dna_molecule.denature()

        for primer in primers:
            primer.denature()


    def anneal(self):
        for dna_molecule in self.dna_molecules:
            for primer in self.primers:
                for genome in self.genomes:
                    primer.bind(genome)
                    # Notify observers about the binding event
                    # self.notify_observers(f"Primer {primer.sequence} tried to bind to genome")


    def extend(self):
        # Simplified extend logic
        pass

    def cycle(self):
        self.denature()
        self.anneal()
        self.extend()


class PCRObserver:
    def update(self, event):
        # Handle event updates for the observer
        print(f"Observed event: {event}")


# class PCRSimulator:
#     def __init__(self, specificity_strategy: PrimerSpecificityStrategy):
#         self._strategy = specificity_strategy
#
#     def set_specificity_strategy(self, strategy: PrimerSpecificityStrategy):
#         self._strategy = strategy
#
#     def run_simulation(self, primer, metagenomic_data):
#         if not self._strategy.test_specificity(primer, metagenomic_data):
#             print("Primer is not specific to the target!")
#             return
#         # Continue with the PCR simulation
#         pass
#
#     def batch_process(self):
#         pass


if __name__ == '__main__':

    ray.init()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='PCR Simulator')

    parser.add_argument('--primers', type=str, required=True, help='Path to the primers FASTA file')
    parser.add_argument('--genomes', type=str, required=True, help='Path to the genomes FASTA file')
    parser.add_argument('--denaturation_temp', type=float, required=True, help='Denaturation temperature in °C')
    parser.add_argument('--annealing_temp', type=float, required=True, help='Annealing temperature in °C')
    parser.add_argument('--extension_temp', type=float, required=True, help='Extension temperature in °C')
    parser.add_argument('--denaturation_time', type=int, required=True, help='Denaturation time in seconds')
    parser.add_argument('--annealing_time', type=int, required=True, help='Annealing time in seconds')
    parser.add_argument('--extension_time', type=int, required=True, help='Extension time in seconds')

    args = parser.parse_args()

    primer_path = "primers.fasta"
    primers = read_primers_from_fasta(primer_path)

    genome_path = "genomes.fasta"

    genome_db_ref = GenomeDatabase.remote()
    genome_db_ref.initialize.remote(genome_path)

    # todo: Pfu, Phi29, Q5 High-Fidelity, Tth, Vent, Deep Vent, Bst, KOD, Specialized Mixes
    taq = Polymerase("Taq", elongation_speed=60, temperature_optimum=75, proofreading_efficiency=0.0, amplifiable_range=(0, 5000))
    phusion = Polymerase("Phusion", elongation_speed=50, temperature_optimum=72, proofreading_efficiency=0.9, amplifiable_range=(0, 10000))

    # Initialize PCR and observer
    pcr = PCR(primers, genome_db_ref)
    observer = PCRObserver()
    pcr.register_observer(observer)

    # Simulate PCR cycles
    num_cycles = 30
    for i in range(num_cycles):
        pcr.cycle()

    ray.shutdown()
