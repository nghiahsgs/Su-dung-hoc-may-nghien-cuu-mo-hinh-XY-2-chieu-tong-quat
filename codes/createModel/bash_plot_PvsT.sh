#deltas=(0.34 1.0)
deltas=(0.2)

#types=(nothing)
#Ls=(12 16 24 32)

types=(xy angle vortex angle_vortex)
Ls=(nothing)
for type in ${types[*]}
  do
    for delta in ${deltas[*]}
      do
        for L in ${Ls[*]}
          do
            echo "current type: ${type} ; delta: ${delta} and ; L: ${L}"
            #echo "python  plot_on_test.py ${delta} ${L}"
            #python  plot_on_test.py ${delta} ${L}
            echo "python  plot_on_test.py ${delta} ${type}"
            python  plot_on_test.py ${delta} ${type}
        done
    done
done
