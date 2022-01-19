import tracemalloc

tracemalloc.start(10)
# 스택 프레임을 최대 10개까지 저장


time1 = tracemalloc.take_snapshot()

for i in range(10000):
    exec('v' + str(i) + ' = ' + str(i))

time2 = tracemalloc.take_snapshot()

stats = time2.compare_to(time1, 'lineno')

for stat in stats[:3]:
    print(stat)
    
stats = time2.compare_to(time1, 'traceback')
top = stats[0]
print('\n'.join(top.traceback.format()))