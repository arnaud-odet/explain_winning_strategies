def display_status_progress(it_wip:int , it_size:int , base_message:str='', percent_step:int = 2, complete_char:str='#', todo_char:str='.'):
    nb_done = (it_wip * 100 // it_size)  //percent_step
    
    message = base_message + (" - "  if base_message != '' else "") + '0% ' + nb_done * complete_char + (100//percent_step - nb_done) * todo_char + ' 100 %'
    print(message, end = '\n' if it_wip == it_size else '\r', flush=True) 
    
def display_status_iterable(it_wip:int , it_size:int , base_message:str='' , it_message:str='' , max_lenght:int=400):
    message = base_message + (" "  if base_message != '' else "")  + f"{it_wip} out of {it_size}" + (" - "  if it_message != '' else "") + it_message
    print(message.ljust(max_lenght), end = '\n' if it_wip == it_size else '\r', flush=True) 