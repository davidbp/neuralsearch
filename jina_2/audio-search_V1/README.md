

# Finding similar audio cuts



The program in `app.py`  uses a `VGGISH` model to encode audio data and search audio data.

To use this code the user is required to first download the `VGGISH` model with the provided scrip shell:

```
./download_model.sh
```



## Basic Usage

The `app.py` can be used in two shell commands:

```shell
python app.py index
```

will index the data found in the repository.

Then:

```shell
python app.py search
```

will read a query audio and present the user with the closest matches.

The query sample is an audio clip from Daphne Koller saying "two plus seven is less than ten". If working correctly the closest item should be `BillGatesSample-3.mp3` which is an audio of Bill Gates saying "two plus seven is less than ten". 

