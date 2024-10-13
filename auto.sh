for size in 16 32 64 128 256 512 1024 2024
do
    for i in {1..5}  # Répéter chaque taille 5 fois
    do
        ./logarithmicScale-result $size >> results.csv
    done
done

