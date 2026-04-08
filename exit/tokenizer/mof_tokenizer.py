import collections
import os
import re
import pkg_resources
from typing import List
from transformers import BertTokenizer
from logging import getLogger

logger = getLogger(__name__)
module_dir = os.path.dirname(__file__)
vocab_path = module_dir + '/vocab_new.txt'
SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]+|[0-9])"

class MOFTokenizer(BertTokenizer):
  def __init__(
      self,
      vocab_file: str = vocab_path,
      # unk_token="[UNK]",
      # sep_token="[SEP]",
      # pad_token="[PAD]",
      # cls_token="[CLS]",
      # mask_token="[MASK]",
      **kwargs):
    
    super().__init__(vocab_file, **kwargs)
    self.max_len = self.model_max_length

    if not os.path.isfile(vocab_file):
      raise ValueError(
          "Can't find a vocab file at path '{}'.".format(vocab_file))
      
    self.vocab = load_vocab(vocab_file)
    
    self.highest_unused_index = max(
        [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
    
    self.ids_to_tokens = collections.OrderedDict(
        [(ids, tok) for tok, ids in self.vocab.items()])
    
    self.basic_tokenizer = BasicSmilesTokenizer(regex_pattern=SMI_REGEX_PATTERN)
    
    self.init_kwargs["max_len"] = self.max_len

    # Special tokens are excluded from meta tokenization.
    self.special_tokens_set = {
        self.pad_token,
        self.cls_token,
        self.sep_token,
        self.mask_token,
        self.unk_token,
    }

    # Sort only non-special tokens from vocab in descending order of length
    # (for longest-match)
    self._meta_tokens_sorted = sorted(
        [t for t in self.vocab_list if t not in self.special_tokens_set],
        key=len,
        reverse=True,
    )

    
  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def vocab_list(self):
    return list(self.vocab.keys())

  def _tokenize(self, text: str):

    # When received the entire MOFID:
    # 1) Before "&&" : SMILES (regex-based)
    # 2) After "&&" : vocab-based
      
    if "&&" not in text:

        smiles_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return smiles_tokens

    # separate "SMILES && meta" 
    smiles_part, meta_part = text.split("&&", 1)

    tokens = []

    # 1) SMILES:  regex
    if smiles_part:
        tokens.extend(self.basic_tokenizer.tokenize(smiles_part))

    # 2) "&&"  (assume "&&" in vocab)
    tokens.append("&&")

    # 3) meta: vocab-based longest-match tokenize
    if meta_part:
        meta_part = meta_part.replace("-", "")
        tokens.extend(self._tokenize_meta(meta_part))

    return tokens

  def _tokenize_meta(self, text: str):
    """
    Tokenize the metastring after '&&' using the longest-match method, matching it to the vocab.
    - Left-to-right scan based on the tokens in vocab_full.txt
    - If no prefix matches, fallback to a single-char (in this case, if not in the vocab, it is mapped to [UNK]).
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
      matched = False

      # Check the longest tokens in vocab first
      for tok in self._meta_tokens_sorted:
        if text.startswith(tok, i):
          tokens.append(tok)
          i += len(tok)
          matched = True
          break

      if not matched:
        # If no vocab token matches, split it into individual characters.
        # (If a character is not in the vocab, it will be mapped to the [UNK] id later.)
        tokens.append(text[i])
        i += 1

    return tokens


    
  def _convert_token_to_id(self, token):
    """
        Converts a token (str/unicode) in an id using the vocab.

        Parameters
        ----------
        token: str
            String token from a larger sequence to be converted to a numerical id.
        """

    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index):
    """
        Converts an index (integer) in a token (string/unicode) using the vocab.

        Parameters
        ----------
        index: int
            Integer index to be converted back to a string-based token as part of a larger sequence.
        """

    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens: List[str]):
    """ Converts a sequence of tokens (string) in a single string.

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.

        Returns
        -------
        out_string: str
            Single string from combined tokens.
        """

    out_string: str = "".join(tokens).strip()
    return out_string

  def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
    """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------

        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        """

    return [self.cls_token_id] + token_ids + [self.sep_token_id]

  def add_special_tokens_single_sequence(self, tokens: List[str]):
    """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.

        """
    return [self.cls_token] + tokens + [self.sep_token]

  def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                           token_ids_1: List[int]) -> List[int]:
    """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

        Parameters
        ----------
        token_ids_0: List[int]
            List of ids for the first string sequence in the sequence pair (A).

        token_ids_1: List[int]
            List of tokens for the second string sequence in the sequence pair (B).
        """

    sep = [self.sep_token_id]
    cls = [self.cls_token_id]

    return cls + token_ids_0 + sep + token_ids_1 + sep

  def add_padding_tokens(self,
                         token_ids: List[int],
                         length: int,
                         right: bool = True) -> List[int]:
    """
        Adds padding tokens to return a sequence of length max_length.
        By default padding tokens are added to the right of the sequence.

        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.

        length: int

        right: bool (True by default)

        Returns
        ----------
        token_ids :
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.

        padding: int
            Integer to be added as padding token

        """
    padding = [self.pad_token_id] * (length - len(token_ids))

    if right:
      return token_ids + padding
    else:
      return padding + token_ids

  def save_vocabulary(
      self, vocab_path: str
  ):  # -> tuple[str]: doctest issue raised with this return type annotation
    """
        Save the tokenizer vocabulary to a file.

        Parameters
        ----------
        vocab_path: obj: str
            The directory in which to save the SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        Returns
        ----------
        vocab_file: :obj:`Tuple(str)`:
            Paths to the files saved.
            typle with string to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        """
    index = 0
    if os.path.isdir(vocab_path):
      vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
    else:
      vocab_file = vocab_path
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(
          self.vocab.items(), key=lambda kv: kv[1]):
        if index != token_index:
          logger.warning(
              "Saving vocabulary to {}: vocabulary indices are not consecutive."
              " Please check that the vocabulary is not corrupted!".format(
                  vocab_file))
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)


class BasicSmilesTokenizer(object):
  """

    Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al. This tokenizer is to be used
    when a tokenizer that does not require the transformers library by HuggingFace is required.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
    >>> tokenizer = BasicSmilesTokenizer()
    >>> print(tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O"))
    ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O']


    References
    ----------
    .. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
            ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
            1572-1583 DOI: 10.1021/acscentsci.9b00576

    """

  def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
    """ Constructs a BasicSMILESTokenizer.
        Parameters
        ----------

        regex: string
            SMILES token regex

        """
    self.regex_pattern = regex_pattern
    self.regex = re.compile(self.regex_pattern)

  def tokenize(self, text):
    """ Basic Tokenization of a SMILES.
        """
    tokens = [token for token in self.regex.findall(text)]
    return tokens


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with open(vocab_file, "r", encoding="utf-8") as reader:
    tokens = reader.readlines()
  for index, token in enumerate(tokens):
    token = token.rstrip("\n")
    vocab[token] = index
  return vocab