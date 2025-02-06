@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        phone = request.form["phone"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode('utf-8')

        new_user = User(email=email, phone=phone, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")
