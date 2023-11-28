from abc import ABC, abstractmethod
import argparse
import primer3
import configparser
from Bio import SeqIO
from Bio.Seq import Seq
from typing import Dict, Tuple, List, Any
import heapq
import subprocess
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import ray
from ray import tune
from ray.tune import Trainable
import os
from gensim.models import Word2Vec
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch


# Parsing the input FASTA file to a dictionary
def parse_fasta_to_dict(fasta_file):
    sequence_dict = {}
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequence_dict[record.id] = str(record.seq)
    return sequence_dict


def parse_fasta_by_amplicons(fasta_file):
    amplicon_dict = {}

    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            # Extract barcode from record.id

            identifier = record.id.split('_')[0]

            # Initialize a sub-dictionary for this identifier if it doesn't exist
            if identifier not in amplicon_dict:
                amplicon_dict[identifier] = {}

            # Add the sequence to the sub-dictionary
            amplicon_dict[identifier][record.id] = str(record.seq)

    return amplicon_dict


# Convert parameters to the correct type
def convert_param(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


# Function to run BLAST for a single primer sequence
def blast_primer(primer_sequence, database):

    result = subprocess.run(
        ['blastn', '-query', '-', '-db', database, '-outfmt',
         '10 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore'],
        input=primer_sequence, encoding='ascii', stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        # If BLAST didn't execute successfully, print the error and return an empty result
        print(f"Error running BLAST: {result.stderr}")
        return []

    # Parse the CSV output from BLAST and return it
    return result.stdout.strip().split('\n')


def parse_blast_output(blast_output):
    reader = csv.DictReader(blast_output, fieldnames=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    return [row for row in reader]


def is_specific(primer_blast_results, params):

    for result in primer_blast_results:
        if float(result['pident']) >= params['IDENTITY_THRESHOLD'] and float(result['evalue']) <= params['EVALUE_CUTOFF']:
            return True  # Primer is specific to at least one sequence
    return False  # No specific alignment found


def output_primers_to_csv(left_primers_info, right_primers_info, left_file_name, right_file_name):
    """
    Output left and right primer information to separate CSV files.

    Parameters:
    left_primers_info (dict): Dictionary containing left primer information.
    right_primers_info (dict): Dictionary containing right primer information.
    left_file_name (str): Name of the output CSV file for left primers.
    right_file_name (str): Name of the output CSV file for right primers.
    """

    # Function to write a single primer info to a CSV file
    # todo: possibly restructure as a standalone function because of the redefinition-related overhead, but most likely it not critical in this case
    def write_primers_to_csv(primers_info, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Primer Sequence', 'Tm', 'GC Content', 'Dimer Partner', 'Dimer Potential',
                                '3-Prime End Complementarity'])

            for primer_seq, dimer_info in primers_info.items():
                if dimer_info:
                    first_partner = dimer_info[0]
                    csvwriter.writerow([primer_seq, first_partner['partner_tm'], first_partner['partner_gc'],
                                        first_partner['partner_sequence'], first_partner['dimer_potential'],
                                        first_partner['three_prime_complementarity']])

                for partner in dimer_info[1:]:
                    csvwriter.writerow(['', '', '', partner['partner_sequence'], partner['dimer_potential'],
                                        partner['three_prime_complementarity']])

    # Write left and right primers to separate CSV files
    write_primers_to_csv(left_primers_info, left_file_name)
    write_primers_to_csv(right_primers_info, right_file_name)


class PrimerSummarizerAlgorithm(ABC):
    @abstractmethod
    def summarize(self, primers: Dict[str, Dict[object]], params) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
        pass


# Implement the Frequency-based Strategy
class FrequencyBasedSummarizer(PrimerSummarizerAlgorithm):
    def summarize(self, primers: Dict[str, Dict[object]], params) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
        # Initialize dictionaries to count the frequency of each primer
        left_primer_freq = {}
        right_primer_freq = {}

        # Iterate over all sets of primer pairs for each sequence
        for primer_set in primers.values():
            for primer_pair in primer_set:
                # Count the left primer frequency
                left_primer = primer_pair['left']
                left_primer_tm = primer_pair['left_tm']
                left_primer_gc = primer_pair['left_gc']

                if left_primer in left_primer_freq:
                    left_primer_freq[left_primer]['freq'] += 1
                else:
                    left_primer_freq[left_primer]['freq'] = 1
                    left_primer_freq[left_primer]['tm'] = left_primer_tm
                    left_primer_freq[left_primer]['gc'] = left_primer_gc

                # Count the right primer frequency
                right_primer = primer_pair['right']
                right_primer_tm = primer_pair['right_tm']
                right_primer_gc = primer_pair['right_gc']

                if right_primer in right_primer_freq:
                    right_primer_freq[right_primer]['freq'] += 1
                else:
                    right_primer_freq[right_primer]['freq'] = 1
                    right_primer_freq[right_primer]['tm'] =  right_primer_tm
                    right_primer_freq[right_primer]['gc'] = right_primer_gc

        # Find the most frequent left and right primers
        most_frequent_left_primers = heapq.nlargest(params['n_most_frequent'], left_primer_freq, key=lambda k: left_primer_freq[k]['freq'])
        most_frequent_right_primers = heapq.nlargest(params['n_most_frequent'], right_primer_freq, key=lambda k: right_primer_freq[k]['freq'])

        # Return the most frequent primer pair with corresponding values of Tm and GC content
        most_frequent_left_primers_info = [(primer, left_primer_freq[primer]) for primer in most_frequent_left_primers]
        most_frequent_right_primers_info = [(primer, right_primer_freq[primer]) for primer in most_frequent_right_primers]

        return most_frequent_left_primers_info, most_frequent_right_primers_info


class ConsensusBasedSummarizer(PrimerSummarizerAlgorithm):
    def summarize(self, primers: Dict[str, Dict[object]], params) -> Dict[str, object]:
        pass


# Define the specificity check algorithm interface
class SpecificityCheckAlgorithm(ABC):
    @abstractmethod
    def specificity_check(self, primers, database, params):
        pass


class SpecificityCheckBLAST(SpecificityCheckAlgorithm):
    def specificity_check(self, primers, database, params):
        # Check specificity for left and right primers
        left_primers_specific = {}
        right_primers_specific = {}

        for left_primer, properties in primers[0]:
            if is_specific(parse_blast_output(blast_primer(left_primer, database)), params):
                left_primers_specific[left_primer] = properties

        for right_primer, properties in primers[0]:
            if is_specific(parse_blast_output(blast_primer(right_primer, database)), params):
                right_primers_specific[right_primer] = properties

        return left_primers_specific, right_primers_specific

class PrimerDimersCheck:
    def is_complementary(self, base1, base2):
        """Check if two bases are complementary."""
        base_pairs = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return base_pairs.get(base1, '') == base2

    def check_3prime_complementarity(self, primer1, primer2, length=4):
        """
        Checks if the last 'length' bases of the 3'-end of two primers are complementary.

        Parameters:
        primer1 (str): Sequence of the first primer.
        primer2 (str): Sequence of the second primer (will be reversed).
        length (int): Number of bases at the 3'-end to check for complementarity.

        Returns:
        bool: True if complementary, False otherwise.
        """

        primer2 = primer2[::-1]

        # Create a translation table for complementarity
        complement = str.maketrans('ATCG', 'TAGC')

        # Check the last 'length' bases for complementarity
        return primer1[-length:].translate(complement) == primer2[:length]

    def sliding_window_dimer_check(self, primer1, primer2, window_size=3):
        """Check for dimer formation using a sliding window approach."""
        max_complementarity = 0
        for i in range(len(primer1) - window_size + 1):
            complementarity = sum(self.is_complementary(base1, base2)
                                  for base1, base2 in zip(primer1[i:i+window_size], primer2[:window_size]))
            max_complementarity = max(max_complementarity, complementarity)
        return max_complementarity

    def check_all_primer_combinations(self, primers_info):
        """
        Check all combinations of primers for potential dimer formation, exclude primers with high dimer potential,
        and return two dictionaries, one for left primers and one for right ones, with remaining primer sequences,
        their 'tm' and 'gc' info and their potential dimer partners with their 'tm' and 'gc' info.
        """
        # Initial primer dimer info
        left_primer_info = {}
        right_primer_info = {}

        # Process left primers against all other primers
        for i, (left_primer_seq, left_primer_data) in enumerate(primers_info[0]):
            for j, (other_primer_seq, other_primer_data) in enumerate(primers_info[0] + primers_info[1]):
                if i != j and self.check_for_dimers(left_primer_seq, other_primer_seq):
                    left_primer_info.setdefault(left_primer_seq, []).append({
                        'partner_sequence': other_primer_seq,
                        'partner_tm': other_primer_data['tm'],
                        'partner_gc': other_primer_data['gc']
                    })

        # Process right primers against all other primers
        for i, (right_primer_seq, right_primer_data) in enumerate(primers_info[1]):
            for j, (other_primer_seq, other_primer_data) in enumerate(primers_info[1] + primers_info[0]):
                if i != j and self.check_for_dimers(right_primer_seq, other_primer_seq):
                    right_primer_info.setdefault(right_primer_seq, []).append({
                        'partner_sequence': other_primer_seq,
                        'partner_tm': other_primer_data['tm'],
                        'partner_gc': other_primer_data['gc']
                    })

        # Identify primers to exclude
        left_to_exclude = self.identify_primers_to_exclude(left_primer_info)
        right_to_exclude = self.identify_primers_to_exclude(right_primer_info)

        # Filter out primers to be excluded
        remaining_left_primers = {seq: info for seq, info in left_primer_info.items() if seq not in left_to_exclude}
        remaining_right_primers = {seq: info for seq, info in right_primer_info.items() if seq not in right_to_exclude}

        return remaining_left_primers, remaining_right_primers

    def check_for_dimers(self, primer1, primer2):
        """
        Check if two primers have potential for dimer formation.
        """
        dimer_potential = self.sliding_window_dimer_check(primer1, primer2)
        three_prime_complementarity = self.check_3prime_complementarity(primer1, primer2)
        return dimer_potential > 0 or three_prime_complementarity

    def identify_primers_to_exclude(self, primer_dimer_info):
        """
        Identify primers with the highest potential for forming dimers to be excluded.
        """
        # Sort primers based on the number of potential dimer partners
        sorted_primers = sorted(primer_dimer_info.items(), key=lambda x: len(x[1]), reverse=True)

        exclusion_threshold = 3
        return [primer_id for primer_id, dimer_partners in sorted_primers if len(dimer_partners) > exclusion_threshold]


# Define the primer design algorithm interface
class PrimerDesignAlgorithm(ABC):
    @abstractmethod
    def design_primers(self, sequences, output_file, common_primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database, num_primers=1, n_most_frequent=5):
        pass


# Design primers using the primer3 library
class Primer3Algorithm(PrimerDesignAlgorithm):
    def design_primers(self, sequences, output_file, common_primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database, num_primers=1, n_most_frequent=5):
        # The code to design primers using the primer3 library
        """
        Select primer targets within a given DNA sequence.

        An example of the Primer3 config file:

        [Primer3Parameters]
        PRIMER_OPT_SIZE = 20
        PRIMER_PICK_INTERNAL_OLIGO = 1
        PRIMER_INTERNAL_MAX_SELF_END = 8
        PRIMER_MIN_SIZE = 18
        PRIMER_MAX_SIZE = 25
        PRIMER_OPT_TM = 60.0
        PRIMER_MIN_TM = 57.0
        PRIMER_MAX_TM = 63.0
        PRIMER_MIN_GC = 20.0
        PRIMER_MAX_GC = 80.0
        PRIMER_MAX_POLY_X = 100
        PRIMER_INTERNAL_MAX_POLY_X = 100
        PRIMER_SALT_MONOVALENT = 50.0
        PRIMER_DNA_CONC = 50.0
        PRIMER_MAX_NS_ACCEPTED = 0
        PRIMER_MAX_SELF_ANY = 12
        PRIMER_MAX_SELF_END = 8
        PRIMER_PAIR_MAX_COMPL_ANY = 12
        PRIMER_PAIR_MAX_COMPL_END = 8

        [SpecificityParameters]
        IDENTITY_THRESHOLD = 95.0
        EVALUE_CUTOFF = 0.01

        """

        primers_for_all_sequences = {}

        for seq_id, sequence in sequences.items():
            # Sequence-specific parameters
            primer_design_params = common_primer_design_params.copy()
            primer_design_params['SEQUENCE_ID'] = seq_id
            primer_design_params['SEQUENCE_TEMPLATE'] = sequence

            primer_results = primer3.bindings.designPrimers(
                primer_design_params,
                {'PRIMER_NUM_RETURN': num_primers}
            )

            # Process the primer results
            primers = set()
            for i in range(num_primers):
                primer_pair = {
                    'left': primer_results[f'PRIMER_LEFT_{i}_SEQUENCE'],
                    'right': primer_results[f'PRIMER_RIGHT_{i}_SEQUENCE'],
                    'left_tm': primer_results[f'PRIMER_LEFT_{i}_TM'],
                    'right_tm': primer_results[f'PRIMER_RIGHT_{i}_TM'],
                    'left_gc': primer_results[f'PRIMER_LEFT_{i}_GC_PERCENT'],
                    'right_gc': primer_results[f'PRIMER_RIGHT_{i}_GC_PERCENT']
                }
                primers.add(primer_pair)

            primers_for_all_sequences[seq_id] = primers


        # Summarizing the primers based on a given algorithm
        summarized_primers = primer_summarizing_algorithm.summarize(primers_for_all_sequences, primer_summarizing_params)

        # Check the specificity of the selected primers using the specified database
        specific_primers = specificity_check_algorithm.specificity_check(summarized_primers, specificity_check_database, primer_specificity_params)

        primer_dimers_check = PrimerDimersCheck()
        remaining_primers = primer_dimers_check.check_all_primer_combinations(specific_primers)

        return remaining_primers


class CustomAlgorithm(PrimerDesignAlgorithm):
    def design_primers(self, sequences, output_file, common_primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database, num_primers=1, n_most_frequent=5):
        # The code for a custom primer design algorithm
        pass


# Implementation of the strategy interface
class Strategy(ABC):
    @abstractmethod
    def design_primers(self, input_file, output_file, primer_design_algorithm: PrimerDesignAlgorithm, primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database):
        pass


# Implementation of concrete strategies for primer design
class TargetedAmpliconSequencingStrategy(Strategy):
    def design_primers(self, input_file, output_file, primer_design_algorithm: PrimerDesignAlgorithm, primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database):
        # Logic for targeted amplicon sequencing
        print("Designing primers for targeted amplicon sequencing...")

        return primer_design_algorithm.design_primers(input_file, output_file, primer_design_params, primer_summarizing_algorithm, primer_summarizing_params, specificity_check_algorithm, specificity_check_database)


class MarkerLociIdentificationStrategy(Strategy):
    @staticmethod
    def sequence_to_kmers(sequence, k):
        return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

    @staticmethod
    def integer_encode_kmer(kmer):
        encoding_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return [encoding_map[nucleotide] for nucleotide in kmer]

    @staticmethod
    def positional_encode_sequence(sequence, k):
        encoded_kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            encoded_kmer = MarkerLociIdentificationStrategy.integer_encode_kmer(kmer)
            # Append positional information
            positional_encoded_kmer = [i] + encoded_kmer  # Prepending the position index
            encoded_kmers.append(positional_encoded_kmer)
        return encoded_kmers

    class _FeatureExtractor(ABC):
        @abstractmethod
        def extract_features(self, sequence):
            pass

    class _KmerFeatureExtractor(_FeatureExtractor):
        def extract_features(self, sequence, k=3):
            return MarkerLociIdentificationStrategy.sequence_to_kmers(sequence, k)

    class _Word2VecFeatureExtractor(_FeatureExtractor):

        def extract_features(self, sequence):
            kmer_extractor = MarkerLociIdentificationStrategy._KmerFeatureExtractor()

            kmers = kmer_extractor.extract_features(sequence)

            model = Word2Vec(sentences=kmers, vector_size=100, window=5, min_count=1, workers=4)

            processed_embeddings = self.process_embeddings(model, kmers)

            return processed_embeddings

        def process_embeddings(self, model, kmers):
            # todo: possibly implement a more advanced logic to process the embeddings

            return [model.wv[kmer].mean(axis=0) for kmer in kmers]

    class _GenomicTransformer(nn.Module):
        def __init__(self, embedding_size, nhead, num_layers, num_classes):
            super().__init__()
            self.transformer = nn.Transformer(d_model=embedding_size, nhead=nhead, num_layers=num_layers)
            self.fc = nn.Linear(embedding_size, num_classes)  # Final classification layer

        def forward(self, x):
            # x: [seq_len, batch_size, embedding_size]
            x = self.transformer(x, x)

            x = x.mean(dim=0)
            x = self.fc(x)
            return x

    class _GenomicTransformer(nn.Module):
        def __init__(self, embedding_size, nhead, num_layers, num_classes):
            super().__init__()
            config = BertConfig(
                hidden_size=embedding_size,
                num_attention_heads=nhead,
                num_hidden_layers=num_layers,
            )
            self.transformer = BertModel(config)
            self.fc = nn.Linear(embedding_size, num_classes)  # Final classification layer

        def forward(self, x):
            # x: [batch_size, seq_len, embedding_size]
    
            # Adjust x to match BERT input dimensions
            x = x.permute(1, 0, 2)  # BERT expects [seq_len, batch_size, embedding_size]

            # Get transformer outputs (last hidden state and attentions)
            outputs = self.transformer(x, output_attentions=True)
            hidden_state = outputs.last_hidden_state
            attentions = outputs.attentions

            # Aggregate over sequence and pass through the classification layer
            aggregated_output = hidden_state.mean(dim=1)
            logits = self.fc(aggregated_output)

            return logits, attentions

    class _TrainGenomicTransformer(Trainable):
        def _setup(self, config):
            # Model and optimizer initialization
            self.model = MarkerLociIdentificationStrategy._GenomicTransformer(config["embedding_size"], config["nhead"], config["num_layers"],
                                            config["num_classes"])
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
            self.criterion = torch.nn.CrossEntropyLoss()

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            # Load and preprocess data
            file_path = config["file_path"]
            train_df, val_df = self._load_and_label_data(file_path)

            # Initialize the label encoder
            label_encoder = LabelEncoder()

            # Create separate DataLoaders for training and validation
            self.train_loader = self.create_data_loader(train_df, config["batch_size"])
            self.val_loader = self.create_data_loader(val_df, config["batch_size"])

        def _create_data_loader(self, df, batch_size):
            # Convert string labels to integers using the instance's label_encoder
            labels = self.label_encoder.fit_transform(df['label'])

            # Convert embeddings list of lists to tensors and pad sequences
            features_list = [torch.tensor(embeddings) for embeddings in df['extracted_features']]
            features_padded = pad_sequence(features_list, batch_first=True, padding_value=0)
            labels = torch.tensor(labels)

            # Create and return a DataLoader
            dataset = TensorDataset(features_padded, labels)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)

         def _train(self):
            num_epochs = self.config.get("num_epochs", 10)

            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                total_train_loss, total_train_accuracy = 0, 0
                for batch in self.train_loader:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_train_loss += loss.item()

                    # Calculate accuracy
                    predicted = torch.argmax(outputs, dim=1)
                    total_train_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

                avg_train_loss = total_train_loss / len(self.train_loader)
                avg_train_accuracy = total_train_accuracy / len(self.train_loader)

                # Validation phase with attention weights collection
                self.model.eval()
                total_val_loss, total_val_accuracy = 0, 0
                with torch.no_grad():
                    for batch in self.val_loader:
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs, attentions = self.model(inputs)  # Get outputs and attention weights
                        loss = self.criterion(outputs, labels)
                        total_val_loss += loss.item()

                        # Calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        total_val_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

                        # Collect attention weights for analysis
                        collected_attention_weights.append(attentions)

                avg_val_loss = total_val_loss / len(self.val_loader)
                avg_val_accuracy = total_val_accuracy / len(self.val_loader)

                # Report metrics for each epoch
                tune.report(train_loss=avg_train_loss, train_accuracy=avg_train_accuracy,
                            val_loss=avg_val_loss, val_accuracy=avg_val_accuracy)

            return {"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy,
                    "val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy}
        
        def analyze_attention_weights(self, attention_weights):
            pass

    def __init__(self, input_file, feature_extractor):
        self._input_file = input_file
        self._feature_extractor = feature_extractor

        # todo: elaborate, connect to CLI parameters
        self._ray_config = {
            "file_path": self._input_file,
            "batch_size": 128,
            "num_epochs": 10,
            "embedding_size": 100,
            "nhead": hp.choice("nhead", [2, 4, 6, 8]),
            "num_layers": 2,
            "num_classes": 2,
            "lr": hp.uniform("lr", 0.0001, 0.01),
        }

    def _load_and_label_data(self, file_path):
        """Loads and labels data from a single FASTA file based on sequence IDs."""
        data = []
        sequence_dict = parse_fasta_to_dict(file_path)
        for seq_id, seq in sequence_dict.items():

            if seq_id.startswith("pathogenic_"):
                pathogenicity = "pathogenic"
            elif seq_id.startswith("non-pathogenic_"):
                pathogenicity = "non-pathogenic"
            else:
                continue

            extracted_features = self._feature_extractor.extract_features(seq)
            data.append({'sequence_id': seq_id, 'extracted_features': extracted_features, 'label': pathogenicity})

            data_df = pd.DataFrame(data)

            # Split data into training and validation sets
            train_df, val_df = train_test_split(data_df, test_size=0.2)

        return train_df, val_df

    def identify_markers(self):
        print("Identifying marker loci...")

        hyperopt_search = HyperOptSearch(space=self._ray_config, metric="accuracy", mode="max")

        # todo: elaborate, connect to CLI parameters
        analysis = tune.run(MarkerLociIdentificationStrategy._TrainGenomicTransformer,
                            config=self._ray_config,
                            num_samples=10,
                            search_alg=hyperopt_search,
                            resources_per_trial={"cpu": 1, "gpu": 1},
                            metric="accuracy",
                            mode="max")

        # Find the best hyperparameters
        best_config = analysis.get_best_config(metric="accuracy", mode="max")

    def design_primers(self, input_file, output_file, primer_design_algorithm: PrimerDesignAlgorithm, primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params, specificity_check_database):
        print("Designing primers for detected marker loci...")


# Context class to use the strategies
class StrategyContext:
    def __init__(self, strategy: Strategy, input_file, output_file, database):
        self._strategy = strategy
        self._input_file = input_file
        self._output_file = output_file
        self._database = database

        if isinstance(self._strategy, TargetedAmpliconSequencingStrategy):
        # self._sequences = parse_fasta_to_dict(input_file)
            self._amplicon_sequences = parse_fasta_by_amplicons(self._input_file)

    def design_primers(self, primer_design_algorithm: PrimerDesignAlgorithm, primer_design_params, primer_summarizing_algorithm: PrimerSummarizerAlgorithm, primer_summarizing_params, specificity_check_algorithm: SpecificityCheckAlgorithm, primer_specificity_params):
        if isinstance(self._strategy, MarkerLociIdentificationStrategy):
            self._strategy.identify_markers()

            # todo:  loop through marker loci, design primers for each of them and output them to a file

        elif isinstance(self._strategy, TargetedAmpliconSequencingStrategy):
            # designed_primers = {}

            for barcode, sequences in self._amplicon_sequences.items():
                primers = self._strategy.design_primers(sequences, self._output_file, primer_design_algorithm, primer_design_params, primer_summarizing_algorithm, primer_summarizing_params, specificity_check_algorithm, primer_specificity_params, self._database)

                # designed_primers[barcode] = primers

                output_primers_to_csv(primers[0], primers[1], self._output_file+barcode+'left_primers.csv', self._output_file+barcode+'right_primers.csv')


if __name__ == "__main__":

    ray.init()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--input_file', required=True, help='The input file to process.')
    parser.add_argument('-o', '--output_file', required=True, default='output.txt', help='The output file to write to.')
    parser.add_argument('-s', '--strategy', required=True, default='amplicon', choices=['amplicon', 'marker'], help='Choose a general strategy - primer design only for the WGS dataset (amplicon) or identification of marker loci in the WGS dataset followed by the primer design for the identified marker loci (marker).')
    parser.add_argument('-a', '--primer_design_algorithm', required=True, default='primer3', choices=['primer3', 'custom'], help='Choose a primer design algorithm.')
    parser.add_argument('-p', '--primer3_parameters', type=str, help='Path to the Primer3 parameters config file.')
    parser.add_argument('-S', '--primer_summarizing_algorithm', required=True, choices=['frequency', 'consensus'], help='An algorithm for summarizing primers.')
    parser.add_argument('-c', '--specificity_check_algorithm', required=True, choices=['blast'], help='An algorithm for primer specificity checking.')
    parser.add_argument('-d', '--database_for_specificity_check', required=True, help = 'A database for checking the specificity of primers.')
    parser.add_argument('-n', '--n_most_frequent', help='A number of the most frequently occurring primers to further work with.')

    args = parser.parse_args()

    if args.strategy == 'amplicon':
        strategy = TargetedAmpliconSequencingStrategy()
    elif args.strategy == 'marker':
        feature_extractor = MarkerLociIdentificationStrategy._Word2VecFeatureExtractor()  # todo: select feature extraction algorithm based on the CLI parameters
        strategy = MarkerLociIdentificationStrategy(args.input_file, feature_extractor)
    else:
        raise ValueError("Unknown strategy specified.")

    context = StrategyContext(strategy, args.input_file, args.output_file, args.database_for_specificity_check)

    if args.strategy == 'marker':
        strategy.identify_markers()

    # Initialize configparser
    config = configparser.ConfigParser()

    if args.primer_design_algorithm == 'primer3':
        if args.primer3_parameters is None:
            raise ValueError("The primer3 config file was not specified.")

        print(f"Primer3 parameters will be loaded from: {args.primer3_parameters}")

        # Read the configuration file
        config.read(args.primer3_parameters)

        primer_design_algorithm = Primer3Algorithm()

        primer_design_params = {param.upper(): config.get('Primer3Parameters', param) for param in config.options('Primer3Parameters')}
        primer_design_params_converted = {key: convert_param(value) for key, value in primer_design_params.items()}
    elif args.primer_design_algorithm == 'custom':
        primer_design_algorithm = CustomAlgorithm()
    else:
        raise ValueError("Unknown primer design algorithm specified.")

    if args.primer_summarizing_algorithm == 'frequency':
        primer_summarizing_algorithm = FrequencyBasedSummarizer()
        primer_summarizing_params = {'n_most_frequent': args.n_most_frequent}
    elif args.primer_summarizing_algorithm == 'consensus':
        primer_summarizing_algorithm = ConsensusBasedSummarizer()
    else:
        raise ValueError("Unknown primer summarizing algorithm specified.")

    if args.specificity_check_algorithm == 'blast':
        specificity_check_algorithm = SpecificityCheckBLAST()
        primer_specificity_params = {param.upper(): config.get('SpecificityParameters', param) for param in config.options('SpecificityParameters')}
        primer_specificity_params_converted = {key: convert_param(value) for key, value in primer_specificity_params.items()}
    else:
        raise ValueError("Unknown primer summarizing algorithm specified.")

    context.design_primers(primer_design_algorithm, primer_design_params_converted, primer_summarizing_algorithm, primer_summarizing_params, specificity_check_algorithm, primer_specificity_params_converted)

    ray.shutdown()
