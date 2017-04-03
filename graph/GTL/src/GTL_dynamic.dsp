# Microsoft Developer Studio Project File - Name="GTL_dynamic" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** NICHT BEARBEITEN **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=GTL_dynamic - Win32 Debug
!MESSAGE Dies ist kein gültiges Makefile. Zum Erstellen dieses Projekts mit NMAKE
!MESSAGE verwenden Sie den Befehl "Makefile exportieren" und führen Sie den Befehl
!MESSAGE 
!MESSAGE NMAKE /f "GTL_dynamic.mak".
!MESSAGE 
!MESSAGE Sie können beim Ausführen von NMAKE eine Konfiguration angeben
!MESSAGE durch Definieren des Makros CFG in der Befehlszeile. Zum Beispiel:
!MESSAGE 
!MESSAGE NMAKE /f "GTL_dynamic.mak" CFG="GTL_dynamic - Win32 Debug"
!MESSAGE 
!MESSAGE Für die Konfiguration stehen zur Auswahl:
!MESSAGE 
!MESSAGE "GTL_dynamic - Win32 Release" (basierend auf  "Win32 (x86) Dynamic-Link Library")
!MESSAGE "GTL_dynamic - Win32 Debug" (basierend auf  "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "GTL_dynamic - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "../bin/"
# PROP Intermediate_Dir "../build/release/GTL_dynamic"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GTL_DYNAMIC_EXPORTS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I "../include/" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GTL_EXPORTS" /YX /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x407 /d "NDEBUG"
# ADD RSC /l 0x407 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386 /out:"../bin/GTL.dll" /implib:"../lib/GTL.lib"
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "GTL_dynamic - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "../bin-debug/"
# PROP Intermediate_Dir "../build/debug/GTL_dynamic"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GTL_DYNAMIC_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /I "../include/" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GTL_EXPORTS" /D "_WINDLL" /D "_AFXDLL" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x407 /d "_DEBUG"
# ADD RSC /l 0x407 /d "_DEBUG" /d "_AFXDLL"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 /nologo /dll /debug /machine:I386 /out:"../bin-debug/GTL.dll" /implib:"../lib-debug/GTL.lib" /pdbtype:sept
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "GTL_dynamic - Win32 Release"
# Name "GTL_dynamic - Win32 Debug"
# Begin Group "Quellcodedateien"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\bellman_ford.cpp
# End Source File
# Begin Source File

SOURCE=.\bfs.cpp
# End Source File
# Begin Source File

SOURCE=.\biconnectivity.cpp
# End Source File
# Begin Source File

SOURCE=.\components.cpp
# End Source File
# Begin Source File

SOURCE=.\debug.cpp
# End Source File
# Begin Source File

SOURCE=.\dfs.cpp
# End Source File
# Begin Source File

SOURCE=.\dijkstra.cpp
# End Source File
# Begin Source File

SOURCE=.\edge.cpp
# End Source File
# Begin Source File

SOURCE=.\embedding.cpp
# End Source File
# Begin Source File

SOURCE=.\fm_partition.cpp
# End Source File
# Begin Source File

SOURCE=.\gml_parser.cpp
# End Source File
# Begin Source File

SOURCE=.\gml_scanner.cpp
# End Source File
# Begin Source File

SOURCE=.\graph.cpp
# End Source File
# Begin Source File

SOURCE=.\maxflow_ff.cpp
# End Source File
# Begin Source File

SOURCE=.\maxflow_pp.cpp
# End Source File
# Begin Source File

SOURCE=.\maxflow_sap.cpp
# End Source File
# Begin Source File

SOURCE=.\min_tree.cpp
# End Source File
# Begin Source File

SOURCE=.\node.cpp
# End Source File
# Begin Source File

SOURCE=.\planarity.cpp
# End Source File
# Begin Source File

SOURCE=.\pq_node.cpp
# End Source File
# Begin Source File

SOURCE=.\pq_tree.cpp
# End Source File
# Begin Source File

SOURCE=.\ratio_cut_partition.cpp
# End Source File
# Begin Source File

SOURCE=.\st_number.cpp
# End Source File
# Begin Source File

SOURCE=.\topsort.cpp
# End Source File
# End Group
# Begin Group "Header-Dateien"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\include\GTL\algorithm.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\bellman_ford.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\bfs.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\biconnectivity.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\bin_heap.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\components.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\debug.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\dfs.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\dijkstra.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\edge.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\edge_data.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\edge_map.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\embedding.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\fm_partition.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\gml_parser.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\gml_scanner.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\graph.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\GTL.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\maxflow_ff.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\maxflow_pp.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\maxflow_sap.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\min_tree.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\ne_map.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\node.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\node_data.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\node_map.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\planarity.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\pq_node.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\pq_tree.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\ratio_cut_partition.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\st_number.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\symlist.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\topsort.h
# End Source File
# Begin Source File

SOURCE=..\include\GTL\version.h
# End Source File
# End Group
# Begin Group "Ressourcendateien"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
