# Tools for cleaning the JW300 parallel corpus

This directory contains the tools used by Universitat d'Alacant for cleaning
the JW300 parallel corpus that can be obtained from http://opus.nlpl.eu/JW300-v1.php.

We applied these tools to obtain a cleaner English-Kyrgyz parallel corpus. We
are releasing them "as is" with the aim of easing the process to other
consortium members.

## Cleaning steps

The script sequentally performs the following actions:

- Downloading JW300 corpus from Opus.
- Detecting the language of each sentence with CLD3 and discarding those sentence pairs whose detected language does not match the expected one.
- Removing sentence pairs that contain Old English words.
- Removing unaligned Bible book references. Very often, the English side of a parallel sentence ends with a reference to a Bible book,
while the Kyrgyz side does not contain it. References are removed from the English side when they are not present in the other language.
This task is carried out by means of regular expressions that may need to be adapted for languages different from Kyrgyz. Example:

> Soon , Satan and these cast - out rebels will be thrown into an abyss for a thousand years . — Revelation 20 : 1 - 3 .  Жакында Шайтан менен асмандан куулган козголоңчу жин - перилер миң жылга туңгуюкка салынышат .

becomes

> Soon , Satan and these cast - out rebels will be thrown into an abyss for a thousand years .  Жакында Шайтан менен асмандан куулган козголоңчу жин - перилер миң жылга туңгуюкка салынышат .

- Removing Unicode non-standard space characters

## Running the cleaning script

- Install the python 3 requirements with: `pip install -r requirements.txt`
- Download and clean corpus with: `bash download-clean-jw.sh TL` where `TL` is the ISO 639-1 of the second language of the desired parallel corpus (the first language is always English).

The resulting corpus will be available in the files `jw300.en-TL.clean.TL` `jw300.en-TL.clean.TL`

## Modifying the cleaning script

For each cleaning step, the script executes a bash function and creates and intermediate file. If you want to skip a step,
just comment out the call to the bash function and change the name of the intermediate file expecteded by the next step.





