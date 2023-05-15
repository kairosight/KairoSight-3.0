# KairoSight 3.0
Python 3.8 software to analyze time series physiological data of optical action potentials.

This project started as a python port of Camat (cardiac mapping analysis tool, PMCID: [PMC4935510](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4935510/)) and is inspired by design cues and algorithms from RHYTHM (PMCID: [PMC5811559](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811559/)) and ImageJ (PMCID: [PMC5554542](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5554542/)). It is important to note that Kairosight 3.0 is currently only available on Windows machines.
 
In order to get up and running with Kairosight 3.0 you will need to complete the following set up steps:
1. First you will need to install Anaconda, which can be found [here](https://docs.anaconda.com/anaconda/install/windows/) or if you have an older operating system [here](https://docs.anaconda.com/free/anaconda/install/old-os/).
2. Select "Just Me - Recommended" when prompted.
3. Clone or download the repository (save the contents to a location that is easy to remember/navigate to).
   a. To find, scroll to the top of this page > locate "Code" (in green) > select the drop-down > locate "Download ZIP" > select "Download ZIP"
   b. Note: If you download the repository, unzip the contents of the zip file (your repository is the folder 'KairoSight-3.0-main')
4. Navigate to your computers search bar and type "Anaconda Prompt"
5. Select the Anaconda Prompt
6. Type 'cd', press the space bar, and then type the directory where you cloned/downloaded the repository (e.g., "OneDrive\Documents\GitHub\KairoSight-3.0-main")
   a. Note: you can copy the file path by finding the directory in your folders, right clicking, and pasting the file path into the anaconda prompt
7. Press 'Enter' on your keyboard
8. Enter the following command: conda env create -f kairosight_env.yml
9. Press 'Enter' on your keyboard
10. When Anaconda has finished installing the environment it should instruct you to run step 11
11. Type 'conda activate kairosight_3-0'
12. Press 'Enter' on your keyboard
13. When anaconda has finished this step, close the 'Anaconda Prompt'
14. Navigate to your computers search bar and type 'Anaconda Navigator'
15. When the application finishes opening, navigate to the drop-down next to 'Application'
   a. Note: the drop-down will most likely say 'base(root)'
16. Switch the drop-down to 'kairosight_3-0'
17. In the save 'Anaconda Navigator' window, find the "Spyder" application
18. Launch the "Spyder" application
19. In the top menu select: Tools -> Preferences
20. Select "IPython Console" on the left hand menu
21. Select the "Graphics" tab and make sure the 'Graphics backend' is set to Qt5
22. Select "Apply" to save any changes, and select "OK" to close the window
23. Navigate to: File -> Open -> 'location of KairoSight-3.0-main' -> src (inside your KairoSight-3.0-main folder) 
24. Select the 'kairosight_retro.py' file
25. When loaded in spyder, select the green play button (in the top menu)
26. KairoSight should now be up and running
