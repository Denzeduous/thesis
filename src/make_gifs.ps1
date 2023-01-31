param (
	[Parameter(Mandatory=$true)][string]$directory
)

$episodes = Get-ChildItem $directory

$episodes | Foreach-Object {
	$dir = "$($directory)\$($_)"
	$images = "$($dir)_Images"

	if (![System.IO.File]::Exists($images)) {
		mkdir $images
	}

	& magick mogrify -format png -colorspace sRGB -density XXX -path "$($images)" "$($dir)/*.svg"
	& magick convert -delay 30 -loop 0 "$($images)/*.png" "$($dir).gif"
}