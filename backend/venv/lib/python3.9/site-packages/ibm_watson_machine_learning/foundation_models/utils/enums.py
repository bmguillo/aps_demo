#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum

__all__ = [
    "ModelTypes",
    "DecodingMethods",
]


class ModelTypes(Enum):
    """Supported foundation models."""
    FLAN_T5_XXL = "google/flan-t5-xxl"
    FLAN_UL2 = "google/flan-ul2"
    MT0_XXL = "bigscience/mt0-xxl"
    GPT_NEOX = 'eleutherai/gpt-neox-20b'
    MPT_7B_INSTRUCT2 = 'ibm/mpt-7b-instruct2'
    STARCODER = 'bigcode/starcoder'
    LLAMA_2_70B_CHAT = 'meta-llama/llama-2-70b-chat'
    LLAMA_2_13B_CHAT = 'meta-llama/llama-2-13b-chat'
    GRANITE_13B_INSTRUCT = 'ibm/granite-13b-instruct-v1'
    GRANITE_13B_CHAT = 'ibm/granite-13b-chat-v1'


class DecodingMethods(Enum):
    """Supported decoding methods for text generation."""
    SAMPLE = "sample"
    GREEDY = "greedy"
