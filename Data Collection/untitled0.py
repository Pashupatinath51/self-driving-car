

def create_stack():
    stack = []
    
    return stack

def length_check(stack):
    
    return len(stack)
print(length_check)
def stack_insert(stack, value):
    
    stack.append(value)
    return stack

def stack_deletion(stack):
    if length_check(stack) == 0:
        return "Empty stack"
        
    return stack.pop()

stack = create_stack()

print('stack :-- ',stack_deletion(stack))


print('stack after pop', stack)

