import subprocess

# Install the requirements
print('imported sub processes')
# subprocess.run(['pip3', 'install', '-r', 'requirements.txt'], check=True)
subprocess.check_call(['pip', 'install', '--no-cache-dir', '-r', 'requirements.txt'])

subprocess.check_call(['pip', 'install', '--upgrade', 'gradio'])

print('done')

#!pip install --no-cache-dir -r requirements.txt
