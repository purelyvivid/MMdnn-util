
# coding: utf-8

from time import ctime, localtime

#python version
from sys import version
print('Python version:',version)

#import ttk
if version[0] == '2':
    from Tkinter import *
    from ttk import *
else: #version[0] == '3':
    from tkinter import *
    from tkinter.ttk import *

from PIL import ImageTk, Image
#from subprocess import call
from try_inference_tf import inference, models


class GUI():
    vb_dict = {}
    def __init__(self, master):
        self.master = master 
        self.row = 0 #for grid
        self.all_comp = []
        self.get_comp = []
        self.vb_name = []  
        self.buildGUI_1()
        #self.show_all_variable()
        
    # --------Mode interface ---------    
    def buildGUI_1(self): #CNN
        self.master.title('build CNN -- load model') 
        # models = ("inception_v3","vgg16","vgg19","resnet","mobilenet","xception","YOLO")
        self.label_1to1_text_combobox("Model", models , width=50 ) 
        self.label_1to1_text_entry(name="ImageUrl", default_text="cat1.jpeg", width=100)
        self.button(self.click_show_img, "Show Image")
        self.button(self.click_inference, "Inference")
        
    def click_show_img(self):
	self.vb_dict = self.generate_variable_dict()
        img_path = self.vb_dict["ImageUrl"]
        self.img_label(path=img_path)
    
    def click_inference(self):
        self.vb_dict = self.generate_variable_dict()
        str_ = inference('cat.jpeg',self.vb_dict["Model"])
        #self.text_text(text=str(str_))
        strs = [""]+str_.split('\n')
        for s in strs:
            self.text_label(text=s)
        
        
    # --------Component Conbination --------- 
    def label_1to1_text_combobox(self, name="", values=("1","2"), default_Chosen=0, width=10):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(StringVar(self.master, values[default_Chosen]))
        self.all_comp.append(Combobox(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1]['values'] = values
        self.all_comp[-1].current(default_Chosen) 
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_float_entry(self, name="", default_float=0.01, width=5):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(DoubleVar(self.master, default_float))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_int_entry(self, name="", default_int=-1, width=10):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(IntVar(self.master, default_int))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_text_entry(self, name="", default_text="", width=35):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(StringVar(self.master, default_text))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1])) 
        #self.all_comp[-1].pack()             
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
            
    def label_1to3_int_entry(self, name="", default_int=[-1,-1,-1], width=10):
        for i in [1,2,3]:
            self.vb_name.append(name.split()[0]+str(i))
        self.all_comp.append(Label(self.master, text=name))
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        for i, int_ in enumerate(default_int):
            self.get_comp.append(IntVar(self.master, int_))
            self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
            #self.all_comp[-1].pack() 
            self.all_comp[-1].grid(row=self.row, column=i+1, sticky=W) #  
        self.row += 1
            
    def label_1to1_bool_checkbutton(self, name="", default_bool=True ,default_text="Yes"):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name))
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(BooleanVar(self.master, default_bool))
        self.all_comp.append(Checkbutton(self.master, text=default_text, variable =self.get_comp[-1], offvalue =False, onvalue =True)); 
        #self.all_comp[-1].pack() 
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #

    def text_label(self, text="OK"):        
        self.all_comp.append(Label(self.master, text=text))         
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #
        
    def text_text(self, text="OK"):        
        text_ = Text(self.master, width=100 , height=30)
        text_.insert(INSERT, text)
        text_.insert(END, "")
        self.all_comp.append(text_) 
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #   
        
    def img_label(self, path="cat1.jpeg"): 
        window = Toplevel()
        #window.geometry('400x400')
        window.title('image')  
        img = ImageTk.PhotoImage(Image.open(path))
        label = Label(window, image = img)
        label.pack()
        window.mainloop()
        
    def button(self, command, text="OK"):        
        self.all_comp.append(Button(self.master, text=text, command = command))         
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #
        
    # ---Show------------------------------------------------------    
    def show_all_variable(self):
        print('show all variable:')
        i=0
        for c in self.get_comp:
            print('{} = {}'.format(self.vb_name[i],c.get()))
            i+=1
              
    def generate_variable_dict(self):
        i=0
        d={}
        for c in self.get_comp:
            d.update({self.vb_name[i]:c.get()})
            i+=1  
        return d   



root = Tk()
my_gui = GUI(root)
root.mainloop() 
    


