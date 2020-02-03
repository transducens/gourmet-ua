#! /bin/bash

function remove_old_english {

	egrep -v " (thy|thou|thee|ye|hath|shalt|hast|brethren|thine|cometh|wilt|doth|maketh|wast|goeth|giveth|dwelleth|knoweth|abideth|spake|seeth|taketh|returneth|sitteth|loveth|doeth|dost|canst|knowest|saidst|lovest|seest|sittest|takest|returnest)[ .,;:?]"
}

function fix_bible_cites {
	sed -r 's:(\([^	]* [0-9]+ [^	]*\)[^	]*)$:\1¿:' | sed -r 's:\xe2\x80\x8b —[^	]* [0-9]+ [^	]*(	[^	]*[^¿])$:\1:' | sed 's:¿$::'
}

function remove_control {
      sed -r 's:\xe2\x80\x8b::g' | sed -r 's:[ ]+: :g'
}

SL="en"
TL=$1
PAIR="en-$TL"

# Download JW300 from Opus
opus_read -d JW300 -s $SL -t $TL -wm moses -w jw300.$PAIR.$SL jw300.$PAIR.$TL

#Detect language with CLD3
paste jw300.$PAIR.$SL jw300.$PAIR.$TL |  python3 filter-cld3.py --lang1 $SL --lang2 $TL   > jw300.$PAIR.cld3.$SL-$TL

#Remove sentences with old English
remove_old_english < jw300.$PAIR.cld3.$SL-$TL > jw300.$PAIR.cld3.oe.$SL-$TL

#Fix non-paralel references to books of the Bible
fix_bible_cites < jw300.$PAIR.cld3.oe.$SL-$TL > jw300.$PAIR.cld3.oe.bible.$SL-$TL

#Remove invisible space character
remove_control < jw300.$PAIR.cld3.oe.bible.$SL-$TL > jw300.$PAIR.cld3.oe.bible.nocontrol.$SL-$TL

cut -f 1 jw300.$PAIR.cld3.oe.bible.nocontrol.$SL-$TL > jw300.$PAIR.clean.$SL
cut -f 2 jw300.$PAIR.cld3.oe.bible.nocontrol.$SL-$TL > jw300.$PAIR.clean.$TL
