import numpy as np
import torch
from scipy.signal import resample
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class whisper_inference_model:
    def __init__(self, new_sample_rate, seconds_per_chunk):
        self.new_sr = new_sample_rate
        self.samples_per_chunk = seconds_per_chunk * self.new_sr
        self.model_name = "openai/whisper-large-v2"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="it",
            task="transcribe",
        )
