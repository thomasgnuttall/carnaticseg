grammar = {
	'mohanam': {
  		'arohana': ['S', 'R2', 'G3', 'P', 'D2', 'S'],
  		'avarohana': ['S', 'D2', 'P', 'G3', 'R2', 'S'],
  	},
  	'gamaka':
  		{'type 1': ['G', 'D']},
  	'jaru': 
  		[('P', 'G')],
  	'phrase':
  		{
  			'pidi': ['dsrg','grs'],
  			'sancara': ['dsrg','grs'],
  			'prayoga': ['dsrg','grs']
  		},
  	'melodic_context': {
  		'svara': [('X', 'Y', 'Y_CHARACTERISTIC'), ('G', 'P', 'stable')],
  		'direction': [('arohana', 'CHARACTERISTIC')]
  		},
}


"""
- kampita gamaka
	- type 1
	- type 2 
	- type 3
	- type ...
- map types uses tempo 


- notation: http://www.shivkumar.org/music/basics/sarali-varisai.htm
	- "," is gap
	- "-"" laya segments (useful with tala)
	- Capital letter = higher octave
	- understand: capital letter, underlined		

- scrape information from shivkumar

- melodic (svara) context:
	- if X comes [after/before] Y then it has CHARACTERISTIC
		- X and Y are svaras
		- CHARACTERISTIC
			- With/without gamaka
			- Held note/not held note
	- if arohana/avorahana then CHARACTERISTIC

- Carnatic wide grammar
	- e.g. such as difficult to oscillate ga when coming from pa since the oscillaton comes from re
	- for two oscillated notes in succesion, the second would be oscillated less
	- In kalpanasvara a performer may chose a certain svara as a reference and sing around that
"""




"""
- no ni
- AUDAVA[five notes]
- pa kampita gamaka


Sancaras from "Ragas in Carnatic Music":
dp,,
gpDpgr-rgpdsd-gpDs-rssddpG-gpD-pds-dsR,
-dsrG-
grgPgrs-dgrgsr-drsD-pgpdsDp-gpDpgr-grpgRS-rsDpDS
"""












