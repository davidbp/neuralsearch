
python -m memory_profiler match.py >> bench_memory_match.txt
mprof run -o match.dat match.py 
mprof plot match.dat -o ./images/match.png

python -m memory_profiler match_batch.py >> bench_memory_match_batch.txt
mprof run -o match_batch.dat match_batch.py 
mprof plot match_batch.dat -o ./images/match_batch.png
