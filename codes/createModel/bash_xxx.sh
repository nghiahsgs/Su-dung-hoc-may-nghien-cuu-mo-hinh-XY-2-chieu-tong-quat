types=(xy angle vortex angle_vortex)
deltas=(0.2 0.34 0.7 1.0)
Ls=(12 16 24 32 40 48 56 64)
for type in ${types[*]}
  do
    for delta in ${deltas[*]}
      do
        for L in ${Ls[*]}
          do
            python plot_on_test.py ${type} ${delta} ${L} 
        done
    done
done
