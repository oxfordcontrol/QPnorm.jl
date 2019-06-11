
using CSV, DataFrames
using Glob

function filter_problems(dir)
	# filter_problems uses $dir/classification.csv to return only problems
	# that satisfy certain criteria.
	# This required .csv file can be created by opening e.g.
	# ftp://ftp.numerical.rl.ac.uk/pub/cutest/sif/mastsif.html
	# with Microsoft Excel, selecting only the names and classification
	# strings and saving it as a csv. 
	# When loaded with CSV it should produce something like
	# │ Row  │ Name     │ Classification      │
	# ├──────┼──────────┼─────────────────────┤
	# │ 1    │ 3PK      │ SBR2-MN-30-0        │
	# │ 2    │ 10FOLDTR │ NOR2-AN-V-0         │
	filepath = string(dir, "classification.csv")
	df = DataFrame()
	if isfile(filepath)
		df = CSV.File(filepath) |> DataFrame
	else
		return glob("*.SIF")
	end

	files = String[]

	n = size(df, 1)
	for i = 1:n
		c = strip(df[:Classification][i])
		file = string(strip(df[:Name][i]), ".SIF")
		n = strip(split(c, '-')[3])
		m = strip(split(c, '-')[4])
		if ((c[1] == 'Q' || c[1] == 'S' || c[1] == 'O') && # Objective quadratic, sum-of-squares or "other"
			c[2] == 'L' && # Linear constraints
			c[3] == 'R' && # Regular first & second derivatives
			c[4] == '2' && # First two derivatives provided
			(n == "V" || parse(Int, n) <= 2000)) # n <= 2000
			# (m == 'V' || parse(Int, m) <= 3000) # m <= 3000

			push!(files, file)
		end
	end
	
	return files
end