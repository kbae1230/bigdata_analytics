def operation(num1, num2, op):
    if op == '*':
        return str(num1*num2)
    elif op == '+':
        return str(num1+num2)
    elif op == '-':
        return str(num1-num2)
    else:
        raise Exception('There is no operation')
    
def makeExpressionList(expression):
    temp = ''
    expression_list = []
    operation_list = []
    for i in a:
        if i.isdigit():
            temp += i
        else:
            expression_list.append(temp)
            expression_list.append(i)
            operation_list.append(i)
            temp = ''
    expression_list.append(temp)
    return expression_list

def makeOperationList(operation_list):
    from itertools import permutations

    return list(permutations(set(operation_list), len(set(operation_list))))

import copy
expression_temp = copy.deepcopy(expression_list)
    # 순열로

def solution(expression):
    expression_list = makeExpressionList(expression)
    operation_list = makeExpressionList(operation_list)
    result = []
    for oper in operation_list:

        for op in oper:
            while op in expression_temp:
                op_index = expression_temp.index(op)
                num2 = expression_temp.pop(op_index+1)
                op = expression_temp.pop(op_index)
                num1 = expression_temp.pop(op_index-1)
                expression_temp.insert(op_index-1,operation(int(num1), int(num2), op)) 

        result.append(abs(int(expression_temp[0])))

        expression_temp = copy.deepcopy(expression_list)

print(max(result))

