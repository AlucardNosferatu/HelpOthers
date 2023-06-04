import guitarpro

song=guitarpro.parse('tab.gp3')
tracks=song.tracks
measures=tracks[0].measures

for no in range(0,len(measures)):
    print('小节：',no+1,'================')
    temp=measures[no].voices[0].beats
#    print('弦:',end='')
#    for index in range(0,len(temp)):
#        if(len(temp[index].notes)>0):
#            print(temp[index].notes[0].string,end='\t')
#    print('')
#    print('品:',end='')
#    for index in range(0,len(temp)):
#        if(len(temp[index].notes)>0):
#            print(temp[index].notes[0].value,end='\t')
#    print('')
    print('音:',end='')
    for index in range(0,len(temp)):
        tempnote=temp[index].notes[0]
        if(len(temp[index].notes)>0):
            print(tempnote.realValue,end='\t')
            if(tempnote.realValue==52 or tempnote.realValue==57):
                tempnote.value=tempnote.value-1

    print('')
    print(' ')

guitarpro.write(song,'tab_after.gp3')