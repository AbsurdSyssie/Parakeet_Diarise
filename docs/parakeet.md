Running the nvidia/parakeet-tdt-0.6b-v3 ASR Model in a GPU Docker Container
Overview of NVIDIA Parakeet TDT 0.6B v3 ASR Model

NVIDIA's Parakeet TDT 0.6B v3 is a 600-million-parameter multilingual speech-to-text model built with NeMo (NVIDIA’s open-source ASR toolkit). It is designed for high-quality transcription and offers features like automatic punctuation/capitalization and accurate word-level & segment-level timestamps in its output. The model can handle long audio (up to ~24 minutes on a high-memory GPU, or up to 3 hours with a modified attention setting) and outputs fully formatted text (with punctuation and casing). Our goal is to deploy this model in a GPU-accelerated Docker container to transcribe .wav audio files and obtain transcripts with timestamps.

In this guide, we’ll build a minimal yet production-ready Docker environment for Parakeet, covering:

Base Image & Environment Setup: Choosing an appropriate CUDA-enabled base image and Python version.

Dependency Installation: Ensuring compatible versions of Python, PyTorch, and NVIDIA NeMo toolkit.

Audio Preprocessing Requirements: Input format (WAV/FLAC, mono, 16 kHz) and normalization steps for reliable transcription.

Avoiding Compatibility Pitfalls: Working around known issues (e.g. Python 3.12 and Lhotse errors) that can cause runtime errors.

Example Inference Script: A transcribe_parakeet.py example showing how to load the model and perform GPU inference with timestamps.

Running the Container: Docker commands with GPU access (--gpus), volume mounts for audio and model cache, and handling Hugging Face model downloads or tokens.

Throughout, we explain why each technical choice is made to ensure a fast, reproducible, and stable transcription workflow.

Selecting the Base Docker Image (CUDA 12.x with Python 3.11)

For GPU acceleration, the container must include NVIDIA’s CUDA runtime libraries. We recommend starting from an official NVIDIA CUDA 12.x base image on Ubuntu 22.04 (Jammy). For example, an image like nvidia/cuda:12.2.0-runtime-ubuntu22.04 provides the CUDA 12.2 runtime and basic OS libraries. This ensures compatibility with recent NVIDIA drivers (CUDA 12.x requires driver version 525+ on the host) and gives us a consistent Linux environment.

Why this base image? It includes the necessary CUDA drivers and cuDNN for GPU inference, without extra bloat. Ubuntu 22.04 is a stable LTS release and supports modern Python versions (making it easier to install Python 3.11, which we'll use). By using the CUDA runtime variant, we get just the libraries needed to run PyTorch/NeMo (the devel image is larger and geared toward compiling code, which isn’t needed for pure inference).

Alternatively, NVIDIA provides ready-to-use containers for NeMo on NGC (e.g., nvcr.io/nvidia/nemo:25.09 was recommended by NVIDIA for Parakeet decoding). Those come pre-installed with the correct PyTorch, NeMo, and dependencies. While convenient, they may include extra development tools and use a specific Python version. For a minimal production image, we’ll assemble only what’s necessary.

GPU Access: Ensure the host has the NVIDIA Container Toolkit installed so Docker can expose the GPU. We will use docker run --gpus all ... when launching the container to enable GPU access. This is vital – if the container can’t see a GPU, PyTorch/NeMo will fall back to CPU. (In fact, if CUDA isn’t available, NeMo logs a warning that decoding will be slower.) Using the --gpus all flag (or older --runtime=nvidia syntax) grants the container permission to use the host GPU devices.

Python, PyTorch, and NeMo – Compatibility and Installation

Choosing Python 3.11 (Not 3.12): We strongly recommend using Python 3.11 in the container, as opposed to the latest 3.12. The reason is that some of NeMo’s dependencies have had issues on Python 3.12. For example, NeMo’s pip install can fail on 3.12 because the pathtools package (pulled in via WandB) didn’t yet support 3.12. More critically, users have encountered runtime errors with NeMo on Python 3.12 due to the Lhotse library (which NeMo uses for data loading). In one case, trying to transcribe audio on Python 3.12 led to a TypeError deep in Lhotse’s dataloader (DynamicCutSampler) due to compatibility issues, causing an "object.init() takes exactly one argument" error. These issues disappear on Python 3.11, which is a well-tested version for NeMo. Thus, we use Python 3.11 for a trouble-free setup (NVIDIA’s own guide uses 3.11 as well).

Installing PyTorch (GPU Build): After setting up Python, install PyTorch with CUDA support before installing NeMo. NeMo does not bundle PyTorch, and if PyTorch is missing, pip might inadvertently install a CPU-only torch. To ensure we get a GPU-enabled PyTorch, we specify the CUDA 12.x wheel. For example, with CUDA 12.1 runtime, one can install PyTorch 2.x via:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


This uses PyTorch’s official wheel repository for CUDA 12.1 (cu121) builds. We choose a PyTorch version known to work with NeMo (PyTorch 2.0 or 2.1 is suitable; those support CUDA 11.8 and 12.1). In the example above, the PyTorch index will provide the latest 2.x version compatible with CUDA 12.1. (In a production environment, you might pin a specific version for stability, e.g. torch==2.1.0+cu121 along with matching torchvision and torchaudio versions.)

Why install PyTorch first? NVIDIA explicitly recommends installing the latest PyTorch before NeMo. This ensures that NeMo’s Python package doesn’t pull in an unintended torch version. By pre-installing the correct CUDA-enabled torch, we guarantee that NeMo will use it and register with the GPU. Skipping this step could lead to NeMo defaulting to a CPU-only torch (resulting in no GPU usage). In short, PyTorch first, NeMo second.

Installing NVIDIA NeMo [ASR] Toolkit: With PyTorch in place, we install NeMo itself. The model card suggests installing NeMo’s ASR collection via pip:

pip install nemo_toolkit['asr']


This will fetch the NeMo toolkit and all ASR-related dependencies. The [asr] extra includes modules needed for speech recognition (Conformer encoder, transducer decoder, etc.) and ensures components like Lhotse (for audio loading) and FastEmit/RNNT libs are installed. Using the nemo_toolkit[asr] package thus pulls in “all necessary dependencies for Parakeet models”, including any custom ops or config libraries.

Technical Note: NeMo’s ASR collection may include compiled components (for example, RNNT beam search or other utilities). These typically come as pre-compiled wheels (or use PyTorch’s implementations) so you likely won’t need a full build toolchain. However, on some platforms you might need a recent GCC/libstdc++ for certain dependencies. Our base Ubuntu 22.04 image provides a modern GCC runtime, but if you encounter errors about GLIBCXX versions, you can install updated libstdc++ (e.g., via apt-get install build-essential or using Conda’s libstdcxx-ng as done in some guides).

Additional System Packages: We include a few system libraries to handle audio processing:

FFmpeg: Installing ffmpeg in the container is recommended. While NeMo can directly read WAV/FLAC, ffmpeg is useful if you need to convert other formats (e.g., MP3) to the required WAV format. It’s a lightweight addition that ensures flexibility in audio handling.

Libsndfile: NeMo (via SoundFile or torchaudio) relies on libsndfile to read WAV/FLAC. On many Linux images, libsndfile1 is preinstalled or bundled with the Python soundfile library. If not, install it (apt-get install libsndfile1) to avoid any audio I/O errors.

Python 3.11 and pip: If our base image doesn’t include Python, we need to install it. For Ubuntu 22.04, we can add the deadsnakes PPA to get Python 3.11 or use the distribution’s python3.11 package. Ensure pip is available (we can use python3.11 -m ensurepip or install python3-pip). All pip operations in the container should explicitly use Python 3.11’s pip to avoid confusion with any system Python.

Summary of Dockerfile Steps: To recap, the Dockerfile (in pseudo-code) would look like:

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python 3.11 and needed system packages
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3-pip ffmpeg libsndfile1 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.x) and NVIDIA NeMo [ASR]
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install nemo_toolkit[asr]

# Copy the transcription script
COPY transcribe_parakeet.py /app/transcribe_parakeet.py

CMD ["python", "/app/transcribe_parakeet.py", "--help"]


Each layer above serves our goal: a specific Python environment, GPU-enabled PyTorch, the NeMo ASR toolkit, and our application code.

Container Run (GPU)

From the repo root:

```bash
docker compose up --build
```

To verify GPU inside the container:

```bash
docker compose run --rm api python3.11 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
PY
```

Cleanup (Docker)

Development runs can leave stopped containers and dangling images. To keep the system clean:

```bash
# Remove stopped containers
docker container prune

# Remove dangling images
docker image prune
```

Use `docker compose run --rm ...` for one-off commands to avoid leaving stopped containers.

Caching Trade-offs

Using pip/model caches speeds up rebuilds and runs, but has downsides:

- Disk usage grows as wheels and model weights accumulate.
- Stale cache risk if versions are not pinned (older wheels may be reused).
- Reproducibility issues if cache hides upstream changes.
- Security risk if outdated cached wheels persist.

Mitigations: pin versions (as we do), periodically prune caches (`docker system prune` or delete `Output/.cache/pip`), and only use `--no-cache` when you need a clean rebuild.

Audio Format Requirements and Preprocessing

For the Parakeet ASR model to work correctly, input audio must be in a specific format. According to NVIDIA, the model expects 16 kHz, mono audio, provided as either WAV or FLAC files. Key points to ensure:

Sample Rate 16 kHz: If your audio is recorded at a different rate (44.1 kHz, 48 kHz, etc.), resample it to 16000 Hz. The model was trained on 16 kHz audio; feeding higher or lower rates can degrade accuracy or even violate the model’s input assumptions. Many audio tools can resample; for example, using ffmpeg:

ffmpeg -i input.mp3 -ar 16000 output.wav


(This converts to 16 kHz. Combine with -ac 1 to ensure mono output.)

Mono Channel (Single Channel): The audio must be mono (one channel). If you have stereo audio, downmix it to mono (e.g., ffmpeg’s -ac 1 flag). NeMo will throw an “input shape mismatch” error if you pass multi-channel audio to a model expecting 1D input. The Parakeet model does not support multi-channel audio – each audio file should contain only one channel. Failing to do this can lead to runtime errors when NeMo constructs its dataloader or features (as some users discovered when trying to transcribe stereo files).

WAV/FLAC format (PCM encoding): Use standard PCM WAV or FLAC files. WAV is simplest for raw audio. Ensure the WAV is in PCM 16-bit or 32-bit float format. The model card notes “16kHz Audio … .wav and .flac audio formats … Monochannel audio”. A 16-bit PCM WAV is typical and will be internally converted to float32 by the NeMo pipeline. If you have MP3/M4A, convert them to WAV first (the NeMo toolkit does not natively decode MP3 without additional codecs – converting externally with ffmpeg is easier).

Normalization (Volume Levels): It’s good practice to have audio at a reasonable volume (not extremely quiet or clipped). The model expects input audio scaled to a typical range (floating-point PCM between -1.0 and 1.0 if normalized). If your WAV is 16-bit PCM, the NeMo toolkit (SoundFile) will load it and scale to float internally. You usually don’t need to manually normalize amplitudes as long as the audio isn’t saturated or silence. Just avoid extremely low gain or heavy clipping. If needed, you can normalize the audio amplitude (e.g., so that peak is at -1 dBFS) using an audio editor or sox, but this is optional – the ASR model is robust to normal variations in volume.

Silence Trimming (optional): Long periods of silence in the audio aren’t harmful, but they will be transcribed as blanks or just increase processing time. You may trim leading/trailing silence if you know the audio contains a lot of dead air. However, do not split the audio into chunks arbitrarily unless you plan to handle segmentation yourself, because Parakeet can handle reasonably long context in one go.

By ensuring the above, we feed the model exactly what it was trained on: 16 kHz single-channel waveforms. This prevents errors such as shape mismatches and maintains transcription accuracy.

Tip – Quick Conversion with ffmpeg: If you have a stereo 44.1 kHz WAV and need to convert, one command can do it:

ffmpeg -i input_stereo.wav -ac 1 -ar 16000 output_mono16k.wav


This will produce a 16 kHz, mono PCM WAV. The -ac 1 ensures one channel, and -ar 16000 resamples. The bit depth will default to 16-bit PCM; to use 32-bit float PCM, you could add -c:a pcm_f32le (but 16-bit is fine as the toolkit will read it into float32 anyway).

Avoiding Common Issues and Ensuring Compatibility

Even with the correct packages installed, there are a few known issues and best practices to keep the setup running smoothly:

Python 3.12 Compatibility Problems: As mentioned, Python 3.12 can introduce issues in the NeMo ASR stack. One known problem was the pathtools dependency failing to install on 3.12, breaking NeMo’s installation. Additionally, an internal change in Python 3.12’s dataclass or init handling caused Lhotse (the audio dataset library) to error out during inference (DynamicCutSampler TypeError). These are low-level issues you might not immediately link to Python version, so it’s important to proactively use Python 3.11 (or 3.10) where these problems do not occur. If you must use Python 3.12, ensure you use the very latest NeMo and Lhotse versions – but the safest course is sticking to 3.11 for now, as we do in this container.

Ensure GPU is Detected: After installing everything, it’s wise to verify that PyTorch sees the GPU. For example, running a quick check inside the container:

import torch
print(torch.cuda.is_available())


should return True (and torch.cuda.get_device_name(0) should show your GPU model). If it returns False, there’s an issue with the environment or how the container is run (e.g., missing --gpus flag or incompatible drivers). In our installation steps, we used the CUDA-specific PyTorch wheels, so as long as the container is launched with GPU access, this should pass. If you accidentally install a CPU-only torch build, NeMo will log that it’s using CPU and you’ll miss out on GPU acceleration.

NeMo Model Download and Caching: The first time you load nvidia/parakeet-tdt-0.6b-v3 with NeMo, it will download a ~2.5 GB checkpoint from Hugging Face. This can take some time and should ideally be done only once. To avoid repeated downloads (and to allow offline reuse), use a shared model cache. By default, NeMo (using huggingface_hub under the hood) will cache the model under ~/.cache/torch/NeMo/.... We can take advantage of this in Docker by mounting a host volume for the cache (more on this in the run instructions section). That way, subsequent container runs find the model locally. In production deployments, you might even bake the model into the image by downloading it at build time (though this makes the image large and is only suitable if you can distribute a 2.5GB+ image). A middle ground is to download on container startup. We will show how to mount a cache directory to achieve persistent storage of the model between runs.

Hugging Face Authentication (if needed): The Parakeet model is licensed CC BY-4.0 and is publicly downloadable, so no authentication token is strictly required. However, for reliability you may consider using a Hugging Face token if you hit bandwidth limits or want to avoid anonymous download restrictions. You can pass a user access token via environment variable. The huggingface_hub library will use HUGGING_FACE_HUB_TOKEN if set. So you can do -e HUGGING_FACE_HUB_TOKEN=<your_token> on docker run to authenticate the download. (You can get a token from your Hugging Face account settings; a read token is sufficient.) This is optional for this model, but we mention it for completeness and for other models that might require an accepted license.

Preventing Inference-Time Delays: In a production scenario, you may not want the first request to pay the model-loading cost. One strategy is to preload the model at container startup. For example, your entrypoint script (or Docker CMD) could run a Python snippet to load the model into memory as a warm-up. This way, the large .nemo file is loaded and the model weights are moved to GPU once during initialization. Subsequent transcription calls will be much faster (since the model is already in memory). If you’re writing a long-running service (e.g., an API server), this approach makes sense. In our simple use-case (CLI transcription of files), we will load the model within the script when needed, but be aware that the first transcription call will include the loading time. By reusing the container or keeping a process alive, you amortize that cost.

Batching and Long Audio Considerations: For very long audio files (dozens of minutes to hours), running in one shot may require a lot of GPU memory. Parakeet v3 can handle ~24 minutes on an 80GB GPU with full attention. On smaller GPUs, you might hit memory limits or slowdowns for long files. NVIDIA provides a method to switch the model to chunked local attention for longer audio (as shown in the model card) – essentially limiting attention context so you can process up to 3 hours with streaming. If you plan to transcribe hour-long files on limited hardware, consider chunking the audio and transcribing in segments (or use the change_attention_model() as in the example). Our example will assume files of manageable length for simplicity. Also note that if you batch multiple audio files in one transcribe() call, the model will process them in parallel (if batch_size is set). This can improve throughput when transcribing many files, but it requires sufficient GPU memory for all batches.

Lhotse Config Errors: NeMo’s integration with Lhotse is mostly behind the scenes for inference, but as seen earlier, unusual errors in Lhotse (like the DynamicCutSampler TypeError) are typically due to version mismatches or Python compatibility. By using the recommended versions (NeMo toolkit as installed via pip and Python 3.11), you sidestep these. If you ever see errors referencing get_lhotse_dataloader_from_config or Lhotse...Dataset, it implies something unexpected in the data pipeline. The common causes are multi-channel audio (addressed by using mono audio) or the Python 3.12 issue already discussed. The resolution in reported cases was to revert to Python 3.11 which we have done. So following the environment guidelines here should avoid Lhotse-related runtime errors altogether.

In summary, our container setup choices (Python 3.11, proper PyTorch installation, mono audio input) are all made to eliminate known failure points. We want the transcription process to be robust: no mysterious crashes or missing libraries when converting audio, no dependency mismatches, and consistent GPU acceleration.

Example: transcribe_parakeet.py – Transcribing Audio with Timestamps

Next, let's write a lightweight Python script that will run inside the container to perform the transcription. We'll call it transcribe_parakeet.py. This script will:

Load the pre-trained Parakeet TDT 0.6B v3 model from NVIDIA’s Hugging Face repository.

Read an input WAV file (path given as a command-line argument).

Use the model to transcribe the audio, including timestamps for each word (and segments).

Print out the transcription and the timestamps.

Here is a sample implementation:

import sys
import nemo.collections.asr as nemo_asr

if len(sys.argv) < 2:
    print("Usage: python transcribe_parakeet.py <audio_file.wav>")
    sys.exit(1)

audio_path = sys.argv[1]

# 1. Load the NVIDIA Parakeet model (will download on first run if not cached)
print("Loading ASR model... (This may take a while the first time)")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
# (The model is large ~2.5GB; ensure you have sufficient GPU memory to load it.)

# 2. Transcribe the audio with timestamps enabled
print(f"Transcribing {audio_path} ...")
results = asr_model.transcribe([audio_path], timestamps=True)

# 3. Output the transcription text
transcription = results[0]  # since we transcribed a single file
print("\nTranscription:\n", transcription.text)

# 4. Output word-level timestamps
print("\nWord-level timestamps:")
for word_info in transcription.timestamp['word']:
    word = word_info['word']
    start = word_info['start']
    end = word_info['end']
    print(f"{word:15s}  start={start:>6.2f}s  end={end:>6.2f}s")


A few things to note in this script:

We use ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3") to fetch the model. This will automatically download and load the model checkpoint if not already cached (NeMo prints log messages during this process). The model is loaded onto the GPU by default if torch.cuda.is_available() returns True.

We call transcribe(..., timestamps=True) on the model. The timestamps=True flag tells NeMo to return time alignment info for characters, words, and segments. By default, word-level and segment-level timestamps are computed for transducer models like Parakeet v3. In the result, transcription.timestamp is a dictionary with keys 'char', 'word', and 'segment'. We access timestamp['word'] to get a list of word timings. Each entry is a dictionary like {'word': 'Hello', 'start': 1.23, 'end': 1.50} indicating when that word was spoken. We iterate over these to print each word with its start and end time. (For segment-level timestamps, one could similarly iterate transcription.timestamp['segment'], where each segment might be a sentence or phrase the model identified, with stamp['segment'] text and times. In this demo, we focus on words.)

The transcript text (transcription.text) already includes punctuation and capitalization, courtesy of the model’s built-in punctuation module. For example, the model might output text like: "Hello, how are you? I am fine." including commas and question marks as appropriate. The word list will include the words without punctuation (punctuation may appear as separate tokens or merged into the timing of the preceding word, depending on how NeMo handles it – typically, punctuation marks get their own timestamp or attached zero-duration token).

Batch Transcription: In our script, we pass a list with a single file. We could pass multiple file paths in the list to transcribe them in one go (the model will create a batch). We can also set a batch_size parameter in transcribe() if we want to control batching explicitly. By default, it will use a batch size of 1 for a list of files, processing one after the other. If you have many short files and a big GPU, increasing the batch size (and passing all file paths) can speed up overall throughput.

GPU Utilization: Because we installed the GPU-enabled PyTorch, asr_model.transcribe() will automatically use the GPU (it moves the model to CUDA and processes the audio tensor on GPU). If you ever wanted to force CPU for debugging, you could call asr_model.to('cpu') before transcribing (not needed here). You can also control the max GPU memory usage by typical CUDA means (not covered here). On a 24GB GPU, this model can transcribe fairly long audio; on a smaller 8GB GPU, you might only manage a few minutes unless you chunk it.

Error Handling: The script does minimal error handling (it will raise exceptions if something goes wrong). In production, you’d catch exceptions (e.g., file not found, out-of-memory, etc.) and handle accordingly. One common runtime issue could be running out of memory for a very large file – in such a case, consider splitting the audio or using streaming mode (NeMo provides a chunked streaming inference script for RNNT models, but that is more complex to integrate).

With transcribe_parakeet.py in place inside the container (we copied it in the Dockerfile), we’re ready to run our container and perform transcriptions.

Running the Docker Container for Transcription

Finally, let’s see how to use our Docker image to transcribe audio. We will run the container with the appropriate flags to grant GPU access, provide the input audio, and persist the model cache. Assuming we built our image and named it parakeet_asr:latest, here’s an example command:

docker run --rm -it \
  --gpus all \
  -v /path/to/local/audio_dir:/data/audio \
  -v /path/to/local/cache_dir:/root/.cache/torch/NeMo \
  -e HUGGING_FACE_HUB_TOKEN=<your_token_if_needed> \
  parakeet_asr:latest \
  python /app/transcribe_parakeet.py /data/audio/sample.wav


Breaking down this command:

--gpus all: This flag ensures Docker makes all NVIDIA GPUs visible inside the container. The container will have access to the CUDA driver and GPU devices. Without this, the container would not see any GPUs (and your transcriptions would run on CPU, or possibly fail if no CPU compatible ops).

-v /path/to/local/audio_dir:/data/audio: We mount a local directory containing audio files into the container (at /data/audio). This is how the container can access input files without baking them into the image. In this example, we assume you have sample.wav in /path/to/local/audio_dir. Inside the container, it will be accessible at /data/audio/sample.wav. We use a volume mount instead of copying files into the image to keep the image generic and to avoid rebuilding for new audio files.

-v /path/to/local/cache_dir:/root/.cache/torch/NeMo: This mounts a local directory as the NeMo model cache. We choose /root/.cache/torch/NeMo because, by default, NeMo will cache models under the current user’s home cache. Our container likely runs as root by default, so root’s cache is used. By mounting it, if the model was downloaded in a previous run, the .nemo file will already be there and NeMo will load from cache instead of downloading again. You can pre-populate this cache by running the container once, or by manually downloading the model. Alternatively, you could mount a higher-level cache (e.g., /root/.cache or set TORCH_HOME). The path we used is specific but ensures we cover the exact directory NeMo uses. After the first run, you should see the model files stored in your host’s cache_dir (around 2.5 GB). On subsequent runs, starting the container with the same cache mount will let NeMo find the model instantly.

-e HUGGING_FACE_HUB_TOKEN=<token>: This sets an environment variable inside the container. If you provide your Hugging Face token (replace <token> with your actual token string), the from_pretrained call will use it to authenticate. This can help avoid any download issues. If you leave this out, it will attempt an anonymous download. Since this model is not gated, anonymous is fine; the token just provides higher rate limits. (If you have already cached the model or baked it in, the token is not needed at all for runtime.)

parakeet_asr:latest: This is the image name (and tag) we built for our container. Use whatever you named your build.

The command at the end: python /app/transcribe_parakeet.py /data/audio/sample.wav – This tells Docker to run our Python transcription script inside the container, and we pass the path to the audio file (as seen inside the container). We could have set the Docker entrypoint to always run the script, but here we explicitly call it for clarity. You can replace sample.wav with any file name present in the mounted audio directory. The --rm -it at the beginning just makes the container run interactively and clean up after finish, which is suitable for one-off transcriptions. In a server scenario, you might run without --rm so the container stays alive.

What to expect on first run: The first time you execute this, if the model isn’t in the cache, you will see log messages as NeMo downloads and extracts the model. It will look something like:

Loading ASR model... (This may take a while the first time)
[NeMo I ...] Downloading nvidia/parakeet-tdt-0.6b-v3 from HuggingFace Hub to /root/.cache/torch/NeMo/NeMo_.../parakeet-tdt-0.6b-v3/...
[NeMo I ...] Instantiating model from pre-trained checkpoint


This can take a minute or two (depending on your internet speed) as ~2.5 GB is fetched. Once downloaded, the model is loaded onto the GPU (which also takes some seconds as 600M parameters are initialized). Then the script will output the transcription result and the timestamps. For example:

Transcription:
 Hello, how are you? I am fine.

Word-level timestamps:
Hello,          start=  0.00s  end= 0.50s
how             start=  0.50s  end= 0.80s
are             start=  0.80s  end= 0.95s
you             start=  0.95s  end= 1.20s
?               start=  1.20s  end= 1.20s
I               start=  2.00s  end= 2.10s
am              start=  2.10s  end= 2.30s
fine            start=  2.30s  end= 2.80s
.               start=  2.80s  end= 2.80s


(The above is an illustrative output – actual times depend on the audio content. Notice punctuation marks like "?" and "." appearing with zero-duration or same start/end, indicating a momentary pause.)

On subsequent runs, with the cache mounted, the model will load from disk without re-downloading. You should still see the [NeMo I ...] Instantiating model from pre-trained checkpoint message, but no download progress. That means it found the cached .nemo file. Transcription of a short audio (a few seconds) is typically very fast on a GPU (the model is optimized for high throughput, capable of real-time or faster transcription). Longer audios will take proportionally longer, but the process is linear – e.g., transcribing a 1-minute clip might take a couple of seconds on a modern GPU.

Why these run settings? We mount volumes instead of embedding data to keep the container stateless and reusable. We enable --gpus all to utilize hardware acceleration. We pass the token env to handle any access issues proactively. All these ensure that our environment remains reproducible: every time we run the container on a machine with an NVIDIA GPU and internet (for first download), we get the same setup and results. By controlling versions (in the Docker build we specified exactly which PyTorch and NeMo we installed, and we use a fixed model name), we avoid drift. This container can thus be deployed in production to consistently transcribe WAV files with high accuracy and speed.
