
for j in 16 32 64 
do
	export FUNCTION="ackley"
	export LABEL=re_${FUNCTION}${j}

	for i in {1..50}
	do
		python evolve_test_function.py -f ${FUNCTION} -d ${j} -g 3500 -l ${LABEL}_${i} -r ${i} -n 100 -o ${LABEL}_p20.data -p 20 -s 12.5 -m 65.536
	done
done

