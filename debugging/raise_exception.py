

# 42/0
# raise Exception('This is the error message')

'''

**************
*            *
*            *
*            *
*            *
**************

'''


def boxPrint(symbol, width, height):
    if len(symbol) != 1:
        raise Exception('"symbol" needs to be a string of length 1.')
    if width < 2 or height < 2:
        raise Exception('"width" and "height" should be greater or equal to 2')

    print(symbol * width)

    for i in range(height - 2):
        print(symbol + (' ' * (width - 2)) + symbol)

    print(symbol * width)

boxPrint('i', 10, 5)

import traceback

try:
    raise Exception('This is the error message')
except:
    errorFile = open('error_log.txt.', 'a')
    errorFile.write(traceback.format_exc())
    errorFile.close()
    print('The traceback info was written error_log.txt')

import os
print(os.getcwd())
path = str(os.getcwd()+ '\\' + 'error_log.txt')
# print(path)
print(open(r'{}'.format(path)).read())
