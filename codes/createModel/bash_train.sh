#types=(xy angle_vortex angle vortex)
types=(xy)
#deltas=(0.34 1.0)
deltas=(0.7)
#Ls=(12 16 24 32 40 48 56 60)
Ls=(12 16 24 32)
#Ls=(12)
#Ls=(56 64)
#Ls=(48 56 60)
for type in ${types[*]}
  do
    for delta in ${deltas[*]}
      do
        for L in ${Ls[*]}
          do
            echo "current type: ${type} ; delta: ${delta} and ; L: ${L}"
            data_dir="/home/nghia/generalized_xy/delta${delta}/hdf5/Delta${delta}/L${L}"
            result_dir="/home/nghia/generalized_xy/codes/model/${type}/Delta${delta}/L${L}"
            #echo "python ml_training.py ${type} ${data_dir} ${result_dir}"
            python ml_training.py ${type} ${data_dir} ${result_dir}
        done
    done
done
