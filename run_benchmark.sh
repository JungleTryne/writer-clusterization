cat benchmarks.txt | while read line
do
   echo "==================== TESTING: $line ===================="
   python3 clusterize.py --cluster-config-path $line
done
