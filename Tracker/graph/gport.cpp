#include "gport.h"


// System specific defines

#if GPORT_MAC
	// From MacTech 4(6) "Comments about PICTs"
	#define picGrpBeg	140
	#define picGrpEnd	141
#endif


// The global port
GBasePort *Port = NULL;


// A sensible default font
GBaseFont::GBaseFont ()
{
    description = "Times-Roman";
    name = "Times-Roman";
    size = 10;
    bold = false;
    italic = true;
}

GPostscriptPort::GPostscriptPort ()
{
    PenWidth = 1;
    DocumentFonts = "";
    Device = devPostscript;
    DisplayRect.SetRect (0, 0, 595-144, 842-144);
	fill_r = 1;
	fill_g = fill_b = 0;
}

void GPostscriptPort::DrawArc (const GPoint &pt, const int radius,
	const double startAngleDegrees, const double endAngleDegrees)
{
	PostscriptStream << "newpath" << endl;
	PostscriptStream << pt.GetX() << " " << -pt.GetY()
    	<< " " << radius
        << " " << (360.0 -startAngleDegrees)
        << " " << (360.0 - endAngleDegrees)
        << " arcn" 		<< endl;
	PostscriptStream << "stroke" 										<< endl;
	PostscriptStream << endl;
}


void GPostscriptPort::DrawLine (const int x1, const int y1, const int x2, const int y2)
{
	//PostscriptStream  << x2 << " " << -y2 << " " << x1 << " " << -y1 << " " << PenWidth << " DrawLine" << endl;
	
    PostscriptStream << "   gsave" 									<< endl;
    PostscriptStream << PenWidth << "   setlinewidth" 							<< endl;
    // We may not always want to set this as it works best with rectangular trees...
//    PostscriptStream << "   2 setlinecap"							<< endl;
//    PostscriptStream << "   0 setgray" 								<< endl;
    //PostscriptStream << "   0.7 setgray" 								<< endl;
	PostscriptStream << fill_r << " " << fill_g << " " <<  fill_b << " setrgbcolor" 										<< endl;
    PostscriptStream << x2 << " " << -y2 << "   moveto" 								<< endl;
    PostscriptStream << x1 << " " << -y1 << "   lineto" 								<< endl;
    PostscriptStream << "   stroke" 								<< endl;
    PostscriptStream << "   grestore" 								<< endl;

}

void GPostscriptPort::DrawCircle (const GPoint &pt, const int radius)
{
	PostscriptStream << "newpath" << endl;
	PostscriptStream << pt.GetX() << " " << -pt.GetY() << " " << radius << " 0 360 arc" 		<< endl;
	PostscriptStream << "stroke" 										<< endl;
	PostscriptStream << endl;
}

void GPostscriptPort::FillCircle (const GPoint &pt, const int radius)
{
	PostscriptStream << "newpath" << endl;
	PostscriptStream << pt.GetX() << " " << -pt.GetY() << " " << radius << " 0 360 arc" 		<< endl;
	//PostscriptStream << "gsave" 										<< endl;
	//PostscriptStream << "0.90 setgray" 										<< endl;
	PostscriptStream << fill_r << " " << fill_g << " " <<  fill_b << " setrgbcolor" 										<< endl;
	
	PostscriptStream << "fill" 										<< endl;
	//PostscriptStream << "grestore" 										<< endl;
	PostscriptStream << endl;
}


void GPostscriptPort::DrawRect (const GRect &r)
{
	PostscriptStream << r.GetLeft() << " " << -r.GetTop() << " moveto" 	<< endl;
	PostscriptStream << r.GetWidth() << " 0 rlineto" 					<< endl;
	PostscriptStream << "0 " << -r.GetHeight() << " rlineto" 			<< endl;
	PostscriptStream << -r.GetWidth() << " 0 rlineto" 					<< endl;
	PostscriptStream << "0 " << r.GetHeight() << " rlineto"				<< endl;
	PostscriptStream << "closepath" 									<< endl;
	PostscriptStream << "stroke" 										<< endl;
	PostscriptStream << endl;
}

void GPostscriptPort::DrawText (const int x, const int y, const char *text)
{
	PostscriptStream  << "(" << text << ") " << x << " " << -y << " DrawText" << endl;
}


void GPostscriptPort::GetPrintingRect (GRect &r)
{
    // A4, with 1" margin
    r.SetRect (0, 0, 595-144, 842-144);
}


void GPostscriptPort::SetCurrentFont (GBaseFont &font)
{
    std::string face = font.GetName();
    if (font.IsBold() || font.IsItalic())
    {
		face += "-";
        if (font.IsBold())
        	face += "Bold";
        if (font.IsItalic())
        	face += "Italic";
    }
/*
	// Duh -- need to do this earlier, perhaps scan the list of
    // fonts already created and output those...
	// Store this font in the list of fonts we need for our document
    int found = DocumentFonts.find_first_of (face, 0);
    if ((found < 0) || (found > DocumentFonts.length()))
    {
    	if (DocumentFonts.length() > 0)
        	DocumentFonts += ", ";
		DocumentFonts += face;
    }
*/
    PostscriptStream << endl;
    PostscriptStream << "/" << face << " findfont" << endl;
    PostscriptStream << font.GetSize () << " scalefont" << endl;
    PostscriptStream << "setfont" << endl;
    PostscriptStream << endl;
}


// Mac
// Win
// Postscript

void GPostscriptPort::SetPenWidth (int w)
{
    PenWidth = w;
    PostscriptStream << w << " setlinewidth" 						<< endl;
    PostscriptStream << endl;
}

void GPostscriptPort::StartPicture (char *pictFileName)
{
    PostscriptStream.open (pictFileName);
        

    // Postscript header
    PostscriptStream << "%!PS-Adobe-2.0" 							<< endl;
    PostscriptStream << "%%Creator: Roderic D. M. Page" 			<< endl;
    PostscriptStream << "%%DocumentFonts: Times-Roman" 		 		<< endl;
    PostscriptStream << "%%Title:" <<  pictFileName 				<< endl;
    PostscriptStream << "%%BoundingBox: 0 0 595 842" 				<< endl; // A4
    PostscriptStream << "%%Pages: 1" 								<< endl;
    PostscriptStream << "%%EndComments" 							<< endl;
    PostscriptStream << endl;

    // Move origin to top left corner
    PostscriptStream << "0 842 translate" << endl;
    PostscriptStream << "72 -72 translate" << endl; // one inch margin

    // Some definitions for drawing lines, etc.

    // Drawline draws text with encaps that project...
    PostscriptStream << "% Encapsulate drawing a line" 				<< endl;
    PostscriptStream << "%    arguments x1 y1 x2 xy2 width" 		<< endl;
    PostscriptStream << "/DrawLine {" 								<< endl;
    PostscriptStream << "   gsave" 									<< endl;
    PostscriptStream << "   setlinewidth" 							<< endl;
    // We may not always want to set this as it works best with rectangular trees...
//    PostscriptStream << "   2 setlinecap"							<< endl;
    PostscriptStream << "   0 setgray" 								<< endl;
    //PostscriptStream << "   0.7 setgray" 								<< endl;
    PostscriptStream << "   moveto" 								<< endl;
    PostscriptStream << "   lineto" 								<< endl;
    PostscriptStream << "   stroke" 								<< endl;
    PostscriptStream << "   grestore" 								<< endl;
    PostscriptStream << "   } bind def" 							<< endl;
    PostscriptStream << endl;

    PostscriptStream << "% Encapsulate drawing text" 				<< endl;
    PostscriptStream << "%    arguments x y text" 					<< endl;
    PostscriptStream << "/DrawText {" 								<< endl;
    PostscriptStream << "  gsave 1 setlinewidth 0 setgray" 			<< endl;
    PostscriptStream << "  moveto" 									<< endl;
    PostscriptStream << "  show grestore" 							<< endl;
	PostscriptStream << "} bind def" 								<< endl;
    PostscriptStream << endl;

}

void GPostscriptPort::EndPicture ()
{
    PostscriptStream << "showpage" 									<< endl;
    PostscriptStream << "%%Trailer" 								<< endl;
    PostscriptStream << "%%end" 									<< endl;
    PostscriptStream << "%%EOF" 									<< endl;
    PostscriptStream.close ();
}



#if GPORT_MAC
// Macintosh
void GMacPort::BeginGroup ()
{
//	::PicComment (picGrpBeg, 0, NULL);
}

void GMacPort::EndGroup ()
{
//	::PicComment (picGrpEnd, 0, NULL);
}
#endif

SVGPort::SVGPort ()
{ 
    fontString = "font-family:Times;font-size:12"; 
    DisplayRect.SetRect (0, 0, 400, 400);
}

void SVGPort::DrawLine (const int x1, const int y1, const int x2, const int y2)
{
    svgStream << "<path style=\"stroke:black;";
    svgStream << ";stroke-width:" << PenWidth;
    svgStream << ";stroke-linecap:square";
    svgStream << "\" "; 
    
    
    svgStream << "d=\"M";
    svgStream << x1 << " " << y1 << " " << x2 << " " << y2 << "\"/>";


//    svgStream << "<g style=\"fill:none;stroke:black\"><path d=\"M";
//    svgStream << x1 << " " << y1 << " " << x2 << " " << y2 << "\"/>";
//    svgStream << "</g>" << endl;
}

void SVGPort::DrawCircle (const GPoint &/*pt*/, const int /*radius*/)
{
/*	PostscriptStream << "newpath" << endl;
	PostscriptStream << pt.GetX() << " " << -pt.GetY() << " " << radius << " 0 360 arc" 		<< endl;
	PostscriptStream << "stroke" 										<< endl;
	PostscriptStream << endl;
*/
}


void SVGPort::DrawRect (const GRect &/*r*/)
{
/*	PostscriptStream << r.GetLeft() << " " << -r.GetTop() << " moveto" 	<< endl;
	PostscriptStream << r.GetWidth() << " 0 rlineto" 					<< endl;
	PostscriptStream << "0 " << -r.GetHeight() << " rlineto" 			<< endl;
	PostscriptStream << -r.GetWidth() << " 0 rlineto" 					<< endl;
	PostscriptStream << "0 " << r.GetHeight() << " rlineto"				<< endl;
	PostscriptStream << "closepath" 									<< endl;
	PostscriptStream << "stroke" 										<< endl;
	PostscriptStream << endl;
*/
}

void SVGPort::DrawText (const int x, const int y, const char *text)
{
    svgStream << "<text x=\"" << x << "\" y=\"" << y << "\" style=\"" << fontString << "\" >"
            << text << "</text>" << endl;
}


void SVGPort::StartPicture (char *pictFileName)
{
    svgStream.open (pictFileName);
    
    svgStream << "<?xml version=\"1.0\" standalone=\"no\"?>" << endl;
    svgStream << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 20010904//EN\" " << endl;
    svgStream << "\"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd\"> " << endl;

    svgStream << "<svg width=\"" << DisplayRect.GetWidth() << "pt\" "
        << "height=\"" << DisplayRect.GetHeight() << "pt\" >" << endl;
        
// <title>test</title>
// <desc>test</desc>
}

void SVGPort::EndPicture ()
{
    svgStream << "</svg>" << endl;
    svgStream.close ();
}


void SVGPort::GetPrintingRect (GRect &r)
{
    r = DisplayRect;
}

void SVGPort::SetCurrentFont (GBaseFont &font)
{
    fontString = "";
    fontString += "font-family:";
    fontString += font.GetName();
	
    char buf[32];
    sprintf (buf, ";font-size:%d", font.GetSize());
    fontString += buf;
    
    if (font.IsItalic())
    {
        fontString += ";font-style:italic";
    }
    if (font.IsBold())
    {
        fontString += ";font-weight:bold";
    }

}



