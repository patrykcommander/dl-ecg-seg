# https://github.com/byschii/ecg-segmentation/tree/main
# code from the github repo mentioned above -> data preprocessing using WFDB library


# type: ignore
import wfdb

from typing import Union,List, Tuple
import numpy as np
import matplotlib.pyplot as plt

LEADS = ['avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
DATA_FOLDER = "./data/physionet.org/files/ludb/1.0.1/data"
SEGMENT_TO_COLOR = {
    'p': 'red',
    'N': 'blue',
    't': 'green',
}
ECG_SIZE = 32*16 # 41*16 # > 32*16 = 512

def get_signal(index:int, as_p_signal:bool=True) -> Union[wfdb.Record, np.ndarray]:
    """
    Get an ECG signal using index (lobachevsky file names are numberes)
    with all 12 leads as a numpy array or a wfdb.record object.
    """

    record = wfdb.rdrecord(DATA_FOLDER + "/" +  str(index))
    assert type(record) is wfdb.Record

    if as_p_signal:
        assert type(record.p_signal) is np.ndarray
        return record.p_signal
        
    return record


def get_annotations(index:int, lead, as_sample=True) -> Union[wfdb.Annotation, np.ndarray]:
    """
    Get an ECG annotation using number (lobachevsky file names are numberes)
    of a specific lead (lobachevsky ECGs consist of 12 leads)
    as a numpy array of indices or a wfdb.annotation object.
    """
    annotations = wfdb.rdann(DATA_FOLDER +"/" + str(index), extension = lead)
    if as_sample:
        return np.array(annotations.sample)
    return annotations


def get_full_ecg(index:int):
    """
    Get full ECG with 12 leads and annotations.
    """
    signal = get_signal(index)
    annotations = [
        get_annotations(index, lead) for lead in LEADS
    ]
    return signal, annotations


def get_single_lead_ecg(index, lead) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a single lead ECG from file index
    and its annotations both as ndarray
    """
    signal = get_signal(index)
    assert type(signal) is np.ndarray
    signal = signal[:, LEADS.index(lead)]

    samples = get_annotations(index, lead)
    assert type(samples) is np.ndarray
    
    return signal, samples


def get_annotations_symbols(index, lead):
    """
    Returns a single lead ECG annotations symbols 
    as a numpy array of annotation characters i.e. ['(', 'N', ')']
    """
    ann = get_annotations(index, lead, as_sample=False)
    return ann.symbol

def paired_annotation_sample_and_symbol(index, lead):
    """
    Returns a tuple of 
    (annotations as indices, annotations as characters)
    """
    annotations_symbols = get_annotations_symbols(index, lead)
    annotations_sample = get_annotations(index, lead)
    return zip(annotations_sample, annotations_symbols)


def get_single_lead_ecg_with_symbols(index, lead):
    """
    Return an ecg signal with 12 leads 
    and its annotations, both as ndarray
    """
    signal = get_signal(index)
    assert type(signal) is np.ndarray
    signal = signal[:, LEADS.index(lead)]

    data = paired_annotation_sample_and_symbol(index, lead)
    
    return signal, np.array(list(data))


def plot_single_lead_ecg(index, lead):
    """
    Plots a single lead ECG with its annotations as indices.
    """
    signal, samples = get_single_lead_ecg(index, lead)

    fig, ax = plt.subplots(figsize = (28, 3))

    ax.plot(signal)
    ax.scatter(samples, signal[samples], c = 'r', marker = 'o')


def plot_signal_with_annotation(index):
    """
    Plots ECG with 12 leads and its annotations as indices.
    """
    signal, samples = get_full_ecg(index)
    # extract sample from annotations
    wfdb.plot_items(signal, samples)


def grouped(itr, n=3):
    """
    Generator that allows to iterate over a list, grouping the elements in groups of n.
    Example:
        >>> list(grouped([1,2,3,4,5,6,7,8,9], n=3))
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    """
    itr = iter(itr)
    end = object()
    while True:
        vals = tuple(next(itr, end) for _ in range(n))
        if vals[-1] is end:
            return
        yield vals


class EGCSignal:
    """
    This class has 4 main purposes:
    1. To store the signal and its annotations
    2. To cut the signal once at the beginning and once at the end
    3. To plot the ECG in different ways
    4. To convert the annotation in a one hot encoding

    Note that doesn't store the entire ECG, but only one lead

    Also has a method to initialize the class without explicitly passing the signal and annotations
    but with the index and lead of the record
    """
    def __init__(self, signal, time_points, symbol, categories = None):
        self.signal:np.ndarray = signal
        self.time_points:np.ndarray = time_points
        self.symbols:list[str] = symbol
        self.symbol_to_category = {
            'N': 0,
            't': 1,
            'p': 2
        }
        self.category_to_symbol = {
            0: 'N',
            1: 't',
            2: 'p'
        }
        self.categories = categories if categories is not None else self.symbols_to_category()
        self._cut_beginning( 550)
        self._cut_end( 3500)

    def __getitem__(self, key):
        return self.signal[key]

    def __len__(self):
        return len(self.signal)

    def _cut_beginning(self, start_point):
        self.signal = self.signal[start_point:]
        self.categories = self.categories[start_point:]

        # now have to check if time_points and symbols are also to cut
        if start_point > self.time_points[0]:
            # get the index of the first time point greater than start_point
            index = np.argmax(self.time_points > start_point)                
            self.time_points = self.time_points[index:]
            self.symbols = self.symbols[index:]

        self.time_points = self.time_points - start_point

        # check the cut point
        if self.categories[0] != -1:
            # if the first symbol is a ')' then i have to prepend a '(' and a letter from self.category_to_symbol
            if self.symbols[0] == ')':
                self.symbols = ['('] + [self.category_to_symbol[self.categories[0]]] + self.symbols
                self.time_points = np.concatenate(([0,1], self.time_points))
            elif self.symbols[0] in self.symbol_to_category:
                # just prepend '('
                self.symbols = ['('] + self.symbols
                self.time_points = np.concatenate(([0], self.time_points))

    def _cut_end(self, end_point):
        self.signal = self.signal[:end_point]
        self.categories = self.categories[:end_point]

        index = self.time_points[self.time_points < self.signal.size].size
        self.time_points = self.time_points[:index]
        self.symbols = self.symbols[:index]

        # check the cut point
        if self.categories[-1] != -1:
            # if the last symbol is a '(' then i have to append a ')' and a letter from self.category_to_symbol
            if self.symbols[-1] == '(':
                self.symbols = self.symbols +  [self.category_to_symbol[self.categories[-1]]] + [')']
                self.time_points = np.concatenate((self.time_points, [self.signal.size - 1, self.signal.size ]))
            elif self.symbols[-1] in self.symbol_to_category:
                # just append ')'
                self.symbols = self.symbols + [')']
                self.time_points = np.concatenate((self.time_points, [self.signal.size]))
 
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize = (28, 3))
        ax.plot(self.signal)

    def plot_with_time_point(self):
        fig, ax = plt.subplots(figsize = (28, 3))
        self.plot(ax)
        ax.scatter(self.time_points, self.signal[self.time_points], c = 'r', marker = 'o')

    def plot_with_segments(self):
        fig, ax = plt.subplots(figsize = (28, 3))
        self.plot(ax)

        for start, symbol, end in grouped(self.time_points, 3):            
            i = np.nonzero(self.time_points == symbol)[0][0]
            current_symbol = self.symbols[i]
            color = SEGMENT_TO_COLOR[current_symbol]
            ax.axvspan(start, end, color=color, alpha=0.4)

    def symbols_to_category(self):
        """
        Converts the symbols list in a numpy array of integers
        same length as the signal.
        Returns a numpy array of categories derived from the 
        symbol_to_category dictionary.  
        """

        # first instantiate an array of -1 same length as the signal
        category = np.full(len(self.signal), -1)
        # now fill the array with the known category
        for section in grouped(self.time_points):
            # unpack the section
            start, peak, end = section
            
            
            # get the category given the peak
            i = np.nonzero(self.time_points == peak)[0][0]
            current_symbol = self.symbols[i]

            category[start:end] = self.symbol_to_category[current_symbol]

        return category

    @staticmethod
    def from_index_and_lead(index, lead):
        return EGCSignal(
        get_signal(index)[:, LEADS.index(lead)],
        get_annotations(index, lead),
        get_annotations_symbols(index, lead) )


def get_class_weights(y: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Returns class weights np.ndarray based on the frequency of each class in the dataset.
    Y is a numpy array of shape (n_examples, n_classes).

    Normalize - If True, returns class weights normalized.
    """
    class_counts = np.zeros(len(np.unique(y))) # numpy array of shape (n_classes, ) storing frequency of each class appearing in the dataset

    for i in range(y.shape[0]):
        unique, counts = np.unique(y[i], return_counts=True)
        occurance_dict = dict(zip(unique, counts))
        for k, v in occurance_dict.items():
            class_counts[k] += v

    total = np.sum(class_counts)

    frequency = class_counts / total
    class_wieghts = 1.0 / frequency

    if normalize:
        class_wieghts = class_wieghts / np.sum(class_wieghts)

    return class_wieghts