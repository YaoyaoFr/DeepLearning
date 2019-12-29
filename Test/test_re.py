import re

text = '<SICE_lambda>0.05</SICE_lambda> \n <SICE_lambda>0.02</SICE_lambda>'
reg = '<SICE_lambda>.*</SICE_lambda>'
match_objs = re.findall(pattern=reg, string=text)
print(match_objs)