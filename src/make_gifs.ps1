Get-ChildItem ".\episodes" | Foreach-Object {
	$dir = "episodes\$($_)"
	Write-Host $dir
	$images = "$($dir)_Images"

	if (![System.IO.File]::Exists($images)) {
		mkdir $images
	}

	& magick mogrify -format png -colorspace sRGB -density XXX -path $images $dir/*.svg
	& magick convert -delay 30 -loop 0 $images/*.png "$($dir).gif"
}