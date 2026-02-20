# Sonata 🎼 🍃 ✂️

A library for music rearrangements (also referred to as cutdowns), written for my master thesis.

The general approach is to create a self-similarity matrix of the beats of a song and then build
a weighted graph of possible paths through these beats.
The target duration of a cutdown will be the length of the path through the graph and the shortest
path will be the best one as determined by your computed features.

🍀 **Note on quality** 🌸
As I used more features for my work than are present in this repo at present time, the resulting rearrangements might vary in quality. Configurations and parameters are therefore to be considered not tuned/optimal.

## Installation

Pre-requisites on macOS:
```sh
brew install portaudio
```

Then
```sh
pipenv sync
```

⚠️  **PLEASE NOTE**: I haven't found a stable set of compatible dependencies that just work. See the Dependency management section for instructions on how to resolve dependency issues you might/will experience.
This also prevented work on containerizing this code.

## Running the code

You can either use the module as a CLI directly or implement your own script. You can use the CLI like this:

```sh
pipenv run python -m sonata <path to song>
```
or if you have your audio in WAV format in the `audio/` dir:

```sh
make run TRACK=myownsong
```

See the CLI help documentation for available arguments to customize how the cutdown is computed.

```sh
pipenv run python- m sonata -h
```

For an example of how to use the library functions see [`example.py`](example.py) or the CLI module.

```sh
make example
```

### Tests

There are some tests providing some semblance of ease of mind

```sh
make test
```

## Library structure

- `analyze`     - Functions for analyzing music
- `cli`         - Code for the CLI and helper functions
- `evaluate`    - Code for evaluating generated cutdowns
- `graph`       - Module creating the graph of beats and graph search
- `metadata`    - Module for managing metadata surrounding a run (an invocation of the tool)

### Custom features

What is missing is an easy way to provide your own functions for computing features from the audio (or anything else for that matter) but adding your own features to the library is fairly straight-forward. The steps are generally:

1. Add your feature to the [`analyze`](sonata/analyze.py) module
1. Compute your features using your function in `calculate_features` and add to return payload
1. Modify the `weights` computation in `analyze_song`
1. Updating the CLI `-w` flag as well

The last step is not as crucial.

## Dependency management

Some hacky weirdness is required to make the libs work together.
Find location of your virtualenv:

```sh
> pipenv run which python
/Users/<username>/<path/to/virtualenv>/bin/python
```
Remember by `export VENV_DIR=/Users/<username>/<path/to/virtualenv>`

### MSAF - `ImportError: cannot import name 'inf' from 'scipy'`
```sh
vim $VENV_DIR/lib/python3.10/site-packages/msaf/pymf/sivm_search.py
```
delete/comment out line 19 (or thereabouts): 
```py
19: from scipy import inf
```

### madmom and Python 3.10 - `AttributeError: module 'collections' has no attribute 'MutableMapping'`

```
pipenv run python
import madmom
```
Check where `madmom` is. Edit processors.py offending line to
```
from collections.abc import MutableSequence
```
Also update the types to `np.float32` and `np.int32` where they show up.

### librosa

Also Librosa...
```
python3.10/site-packages/librosa/beat.py", line 507, in __trim_beats
    smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.windows.hann(5), "same")
AttributeError: module 'scipy.signal' has no attribute 'hann'
```
scipy.signal.hann -> scipy.signal.windows.hann

when using annotated beats in segmenting:

5/lib/python3.10/site-packages/librosa/beat.py", line 507, in __trim_beats
    smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.hann(5), "same")

# References

If you want to learn more about this approach:

- ”Detection of cut-points for automatic music rearrangement”. _Daniel Stoller, Vincent Akkermans, and Simon Dixon_. In: 2018 IEEE 28th International Workshop on Machine Learning for Signal Processing
- ”Music Rearrangement Using Hierarchical Segmentation” _Christos Plachouras and Marius Miron_. In: 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- _My thesis_ ✍🏻
