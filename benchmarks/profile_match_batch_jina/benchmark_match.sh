
#######################
# Profile computation #
#######################

# profile match.py
kernprof -l match.py 
python -m line_profiler match.py.lprof >> bench_compute_match.txt

# profile match_batch.py
kernprof -l match_batch.py 
python -m line_profiler match_batch.py.lprof >> bench_match_batch.txt

##################
# profile memory #
##################

# profile match.py
python -m memory_profiler match.py >> bench_memory_match.txt
mprof run -o match.dat match.py 
mprof plot match.dat -o ./images/match.png

# profile match_batch.py
python -m memory_profiler match_batch.py >> bench_memory_match_batch.txt
mprof run -o match_batch.dat match_batch.py 
mprof plot match_batch.dat -o ./images/match_batch.png
