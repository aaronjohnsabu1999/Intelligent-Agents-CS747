((i=0))
for instance in "../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt"
do
    for algorithm in "thompson-sampling-with-hint"
    do
        for horizon in 100 400 1600 6400 25600 102400
        do
            for seed in {0..49}
            do
                echo -ne "\\r $instance : $algorithm : $horizon : $seed : iteration $i                "
                python3 bandit.py --instance $instance --algorithm $algorithm --randomSeed $seed --epsilon 0.02 --horizon $horizon >> outputDataT2.txt
                ((i=i+1))
            done
        done
    done
done 