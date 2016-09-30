
#ifndef __Q_OPENGL_TRACKBALL_WIDGET_H__
#define __Q_OPENGL_TRACKBALL_WIDGET_H__

#include <QOpenGLWidget>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QMatrix4x4>

class QOpenGLTrackballWidget : public QOpenGLWidget
{
public:
	explicit QOpenGLTrackballWidget(QWidget *parent = 0);
	~QOpenGLTrackballWidget();

	void setWeelSpeed(float weel_speed){ weelSpeed = weel_speed; }
	void setPosition(float x, float y, float z){ position = QVector3D(x, y, z); }

	void setPerspective(float fov_y, float near_plane, float far_plane)
	{ 
		fovy = fov_y;
		nearPlane = near_plane;
		farPlane = far_plane;
	}

protected:
	virtual void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
	virtual void mousePressEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void mouseMoveEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void mouseReleaseEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void wheelEvent(QWheelEvent* e) Q_DECL_OVERRIDE;
	virtual void timerEvent(QTimerEvent *e) Q_DECL_OVERRIDE;
	virtual void resizeGL(int w, int h) Q_DECL_OVERRIDE;

	QBasicTimer timer;
	QVector2D mousePressPosition;
	QVector3D rotationAxis;
	qreal angularSpeed;
	QQuaternion rotation;
	QVector3D position;
	float weelSpeed;
	float fovy;
	float nearPlane;
	float farPlane;
	QMatrix4x4 projection;
};




#endif // __Q_OPENGL_TRACKBALL_WIDGET_H__
