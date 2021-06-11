
# Stop zookeeper
kubectl delete -f zookeeper.yaml

# Stop kafka
kubectl delete -f kafka.yaml

# Stop spark
kubectl delete -f spark.yaml
