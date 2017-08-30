if (( $# != 1 ))
then
	echo "Usage: runme.py <num_of_times>"
	exit 1
fi
echo "first param = $1"
touch byzantine_stats.txt
for (( i=0; i < $1; i++ ))
do
	python3 bgd.py > outfile
	echo "done outfile, i = $i"
	tail -1 outfile >> byzantine_stats.txt
	rm outfile
	#y
done
exit 0

