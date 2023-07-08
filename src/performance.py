class Performance:
	"""
	Object to handle all performance information
	"""
	def __init__(self):
		self.tonic = None
		self.name = None
		self.performer = None
		self.raga = None
		self.pitch_track = None
		self.vocal = None
		self.sr = None
		self.mridangam_left = None
		self.mridangam_right = None
		self.loudness = None
		self.yticks_dict = None
		self.annotations = None
		self.track_id = None
	
	def initialise(self, stuff):
		pass

	def load(self, dir):
		pass

	def write(self, dir):
		pass

