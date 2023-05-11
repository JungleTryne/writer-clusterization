cat benchmarks.txt | while read line
do
   python3 clusterize.py --cluster-config-path $line
done
