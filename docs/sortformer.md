NVIDIA NeMo
To train, fine-tune or perform diarization with Sortformer, you will need to install NVIDIA NeMo[6]. We recommend you install it after you've installed Cython and latest PyTorch version.

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]

🚀 Quick Start: Run Diarization Now
Here is a short example script that loads the model, runs diarization on a WAV file, and prints the results:

from nemo.collections.asr.models import SortformerEncLabelModel
diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
diar_model.eval()

diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

predicted_segments = diar_model.diarize(audio=["/path/to/your/audio.wav"], batch_size=1)

for segment in predicted_segments[0]:
    print(segment)

How to Use this Model
The model is available for use in the NeMo Framework[6], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

Loading the Model
from nemo.collections.asr.models import SortformerEncLabelModel

# load model from Hugging Face model card directly (You need a Hugging Face token)
diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")

# If you have a downloaded model in "/path/to/diar_streaming_sortformer_4spk-v2.nemo", load model from a downloaded file
diar_model = SortformerEncLabelModel.restore_from(restore_path="/path/to/diar_streaming_sortformer_4spk-v2.nemo", map_location='cuda', strict=False)

# switch to inference mode
diar_model.eval()

Input Format
Input to Sortformer can be an individual audio file:

audio_input="/path/to/multispeaker_audio1.wav"

or a list of paths to audio files:

audio_input=["/path/to/multispeaker_audio1.wav", "/path/to/multispeaker_audio2.wav"]

or a numpy array (single or list):

import numpy as np
audio_input = np.random.randn(16000 * 10).astype(np.float32)  # 10 sec at 16kHz
# or a list of arrays
audio_input = [audio_array1, audio_array2]
diar_model.diarize(audio=audio_input, batch_size=2, sample_rate=16000)

Note: When using numpy arrays, you MUST specify a correct sample_rate in diar_model.diarize() function. Default sample_rate is 16000.

or a jsonl manifest file:

audio_input="/path/to/multispeaker_manifest.json"

where each line is a dictionary containing the following fields:

# Example of a line in `multispeaker_manifest.json`
{
    "audio_filepath": "/path/to/multispeaker_audio1.wav",  # path to the input audio file 
    "offset": 0, # offset (start) time of the input audio
    "duration": 600,  # duration of the audio, can be set to `null` if using NeMo main branch
}
{
    "audio_filepath": "/path/to/multispeaker_audio2.wav",  
    "offset": 900,
    "duration": 580,  
}

Setting up Streaming Configuration
Streaming configuration is defined by the following parameters, all measured in 80ms frames:

CHUNK_SIZE: The number of frames in a processing chunk.
RIGHT_CONTEXT: The number of future frames attached after the chunk.
FIFO_SIZE: The number of previous frames attached before the chunk, from the FIFO queue.
UPDATE_PERIOD: The number of frames extracted from the FIFO queue to update the speaker cache.
SPEAKER_CACHE_SIZE: The total number of frames in the speaker cache.
Here are recommended configurations for different scenarios:

Configuration	Latency	RTF	CHUNK_SIZE	RIGHT_CONTEXT	FIFO_SIZE	UPDATE_PERIOD	SPEAKER_CACHE_SIZE
very high latency	30.4s	0.002	340	40	40	300	188
low latency	1.04s	0.093	6	7	188	144	188
For clarity on the metrics used in the table:

Latency: Refers to Input Buffer Latency, calculated as CHUNK_SIZE + RIGHT_CONTEXT. This value does not include computational processing time.
Real-Time Factor (RTF): Characterizes processing speed, calculated as the time taken to process an audio file divided by its duration. RTF values are measured with a batch size of 1 on an NVIDIA RTX 6000 Ada Generation GPU.
To set streaming configuration, use:

diar_model.sortformer_modules.chunk_len = CHUNK_SIZE
diar_model.sortformer_modules.chunk_right_context = RIGHT_CONTEXT
diar_model.sortformer_modules.fifo_len = FIFO_SIZE
diar_model.sortformer_modules.spkcache_update_period = UPDATE_PERIOD
diar_model.sortformer_modules.spkcache_len = SPEAKER_CACHE_SIZE
diar_model.sortformer_modules._check_streaming_parameters()

Getting Diarization Results
To perform speaker diarization and get a list of speaker-marked speech segments in the format 'begin_seconds, end_seconds, speaker_index', simply use:

predicted_segments = diar_model.diarize(audio=audio_input, batch_size=1)

If you want to use

To obtain tensors of speaker activity probabilities, use:

predicted_segments, predicted_probs = diar_model.diarize(audio=audio_input, batch_size=1, include_tensor_outputs=True)

Note that if you are feeding a list of numpy arrays, you MUST provide the sample_rate in integer format.

predicted_segments, predicted_probs = diar_model.diarize(audio=[np_array1, np_array2], batch_size=2, sample_rate=16000)

🔬 For more detailed evaluations (DER)
If you need to perform a comprehensive evaluation and calculate the Diarization Error Rate (DER) across different parameter settings, use the NeMo example script e2e_diarize_speech.py. This script allows you to test the streaming behavior of the model by adjusting key parameters like chunk_len, fifo_len, and spkcache_update_period.

python ${NEMO_ROOT}/examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py \
    model_path="/path/to/diar_sortformer_4spk_v1.nemo" \
    dataset_manifest="/path/to/diarization_manifest.json" \
    batch_size=1 \
    spkcache_len=188 \
    spkcache_update_period=300 \
    fifo_len=40 \
    chunk_len=340 \
    chunk_right_context=40

Input
This model accepts single-channel (mono) audio sampled at 16,000 Hz.

The actual input tensor is a Ns x 1 matrix for each audio clip, where Ns is the number of samples in the time-series signal.
For instance, a 10-second audio clip sampled at 16,000 Hz (mono-channel WAV file) will form a 160,000 x 1 matrix.
Output
The output of the model is an T x S matrix, where:

S is the maximum number of speakers (in this model, S = 4).
T is the total number of frames, including zero-padding. Each frame corresponds to a segment of 0.08 seconds of audio.
Each element of the T x S matrix represents the speaker activity probability in the [0, 1] range. For example, a matrix element a(150, 2) = 0.95 indicates a 95% probability of activity for the second speaker during the time range [12.00, 12.08] seconds.