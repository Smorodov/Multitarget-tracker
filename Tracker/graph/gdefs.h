#ifndef GDEFH
#define GDEFH

// Determine which platform we are building for



#if __BORLANDC__  // Borland specific options
	#define GPORT_WINDOWS 1 						// Windows
	#define GPORT_MAC	  0
#endif

#ifdef __MWERKS__ //  Metrowerks specific options
	#ifdef __INTEL__
		#define GPORT_WINDOWS 1 					// Windows
		#define GPORT_MAC	  0
  		#define __WIN32__							// MetroWerks only supports Win32
	#endif /* __INTEL__ */

	#ifdef macintosh								// MacOS
	#ifdef __WXMAC__
		#define USE_WXWINDOWS 1						// wxWindows
		#define GPORT_MAC     0	 					
		#define GPORT_WINDOWS 0
	#else	
		#define GPORT_MAC     1	 					// Macintosh
		#define GPORT_WINDOWS 0
	#endif
	#endif /* macintosh */
#endif

#ifdef __GNUC__
		#define GPORT_MAC     0	 					// Assume gcc implies X windows
		#define GPORT_WINDOWS 0
#endif

#endif

