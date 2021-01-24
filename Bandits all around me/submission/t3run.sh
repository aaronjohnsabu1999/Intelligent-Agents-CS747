((i=0))
for instance in "../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt"
do
    for epsilon in 0.001 0.05 0.8
    do
        for horizon in 102400
        do
            for seed in {0..49}
            do
                for algorithm in "epsilon-greedy"
                do
                    echo -ne "\\r $instance : $algorithm : $seed : $epsilon : $horizon : iteration $i                "
                    python3 bandit.py --instance $instance --algorithm $algorithm --randomSeed $seed --epsilon $epsilon --horizon $horizon >> outputDataT3.txt
                    ((i=i+1))
                done
            done
        done
    done
done