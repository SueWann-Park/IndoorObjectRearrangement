def convert_xy():
    f = open('gsr_1.txt', 'r')
    result = open('gsr_3.txt', 'w')
    for line in f.readlines():
        item = line.split(' ')
        if(len(item) == 6):
            result.write(item[0] + ' ' + item[2] + ' ' + item[1] + ' ' +
                             item[3] + ' ' + item[4] + ' ' + item[5])
    f.close()
    result.close()

if( __name__ == '__main__'):
    convert_xy()