from itertools import permutations
import copy


class MaximumOperation:
    
    def __init__(self):
        self.expression_list = []
        self.operation_list = []

        
    def makeExpressionList(self, expression):
        temp = ''

        for i in expression:
            
            if i.isdigit():
                temp += i
            else:
                self.expression_list.append(temp)
                self.expression_list.append(i)
                self.operation_list.append(i)
                temp = ''
                
        self.expression_list.append(temp)
        return self.expression_list


    def makeOperationList(self):

        return list(permutations(set(self.operation_list), len(set(self.operation_list))))
    
       
    def operation(self, num1, num2, op):
        if op == '*':
            return str(num1*num2)
        elif op == '+':
            return str(num1+num2)
        elif op == '-':
            return str(num1-num2)
        else:
            raise Exception('There is no operation')


    def solution(self, expression):
        
        self.expression_list = self.makeExpressionList(expression)
        self.operation_list = self.makeOperationList()
 
        expression_temp = copy.deepcopy(self.expression_list)
        
        result = []
        for oper in self.operation_list:

            for op in oper:
                while op in expression_temp:
                    op_index = expression_temp.index(op)
                    num2 = expression_temp.pop(op_index+1)
                    op = expression_temp.pop(op_index)
                    num1 = expression_temp.pop(op_index-1)
                    expression_temp.insert(op_index-1, self.operation(int(num1), int(num2), op)) 

            result.append(abs(int(expression_temp[0])))

            expression_temp = copy.deepcopy(self.expression_list)

        return max(result)
    

if __name__ == '__main__':
    # expression = "100-200*300-500+20"
    expression = "100-200*300-500+20"
    maximumOperation = MaximumOperation()
    result = maximumOperation.solution(expression)
    print(result)
    
# expression = re.split('([-+*])',expression) 로도 가능
# eval() 안에 str 이어도 계산
